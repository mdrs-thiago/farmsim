"""
build_structure_raster.py

The STRUCTURE channel: bed (crop plot) vs corridor (drivable, between
plots) vs void (outside the surveyed block).

Two independent ways to build it, selected by whether --plot-polygons is
given:

  Plot-polygon mode (--plot-polygons + --plot-lattice + --topdown-meta):
  directly rasterizes annotate_plots.py's annotated plot polygons as BED,
  the annotated block boundary (plot_lattice's corners) as the surveyed
  footprint, and everything inside that footprint but outside a plot as
  CORRIDOR. Strictly more accurate once real annotated boundaries exist
  (Phase A) -- no relief threshold to mistune, no false positives from
  terrain/mesh noise. This is now the default whenever plot annotations
  are available.

  Relief mode (default when --plot-polygons is omitted, kept as a
  fallback/cross-check for surveys without full plot annotation): purely
  geometric row/corridor detection from terrain relief, independent of
  color/hue and thus stable across lighting and growth stage in a way
  build_label_raster.py's vegetation channel never was. "Row"
  (segment_rows.py) means the drivable bare-soil corridor BETWEEN crop
  beds -- physically a VALLEY relative to the beds either side of it. This
  mode finds that valley directly from the baked mesh's own height field,
  instead of from hue.

Method: rasterize height onto the same world-XY grid build_label_raster.py
uses (raster_common.build_output_grid + load_height_lookup), then DETREND
it -- subtract a wide, masked local-mean low-pass from the raw height --
so what's left (the residual) is relief RELATIVE to the immediate
neighborhood, not absolute elevation. This matters because the field
itself has real relief/slope (documented in build_label_raster.py's own
notes: ~0.2m even along one fairly flat row) that a fixed absolute
threshold can't tell apart from an actual corridor. A cell whose residual
drops measurably below its local neighborhood is a corridor; everything
else in-footprint is a bed.

Detrending happens on the RASTERIZED grid, not the raw ~198k-vertex point
cloud: scipy.ndimage.uniform_filter is O(1) per output pixel regardless
of window size (a sliding-sum algorithm), whereas a KD-tree radius-ball
local-mean over the point cloud at field-scale radius (e.g. 10m, ~3000
neighbors/vertex on this survey's ~18cm vertex spacing) would be far more
expensive for no accuracy benefit at this resolution.

Default --detrend-radius-m (10.0) is sized against this field's actual
measured bed/corridor pitch (~6.3-6.4m between corridor centerlines, from
config/rows_20250425.yaml) -- about 3x that period, so the low-pass
averages across multiple full corridor/bed cycles (a corridor doesn't get
partially absorbed into its own local trend) while staying finer than the
whole field (still tracking genuine field-wide undulation). Pitch will
differ per survey/growth-stage layout, hence a flag, not a constant.

A per-pixel depth threshold alone false-positives on any small local dip
that crosses it for reasons that have nothing to do with a real corridor
(real terrain/mesh noise, a slightly-lower patch of otherwise-normal
crop) -- confirmed against real output: areas that are actually crop got
classified as corridor purely because their residual happened to be low.
Fix: after thresholding, keep a corridor cell only if it's part of a
genuinely long, continuous run along the row direction (--corridor-min-
length-m, --row-axis) -- same "long corridor survives, short blob
doesn't" distinction segment_rows.py's extract_row_corridors already
makes on the birdseye photo (there via rotate-then-open on a perspective
capture; here directly on the raster grid via binary_opening, since this
is already an orthographic, world-XY-aligned projection with a known row
axis -- confirmed this survey's rows run at ~0.0 deg from +X, see
config/rows_20250425.yaml -- so no rotation/warp step is needed first).

Usage (relief mode, unchanged):
  python farm/tools/build_structure_raster.py \
      --georef farm/fields/20250425/georef.json \
      --mesh farm/fields/20250425/label_mesh.glb \
      --resolution 0.05 \
      --detrend-radius-m 10.0 \
      --corridor-depth-m 0.03 \
      --out farm/fields/20250425/structure_raster.png

Usage (plot-polygon mode):
  python farm/tools/build_structure_raster.py \
      --georef farm/fields/20250425/georef.json \
      --mesh farm/fields/20250425/label_mesh.glb \
      --topdown-meta farm/fields/20250425/topdown.yaml \
      --plot-polygons config/plot_polygons_20250425.geojson \
      --plot-lattice config/plot_lattice_20250425.json \
      --resolution 0.05 \
      --out farm/fields/20250425/structure_raster.png
"""

import argparse
import json

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import binary_opening, uniform_filter

from farmsim_farm import raster_common as rc
from exg_classify import load_plot_polygons, rasterize_plot_footprint, rasterize_polygon
from farmsim_farm.palettes import STRUCT_BED, STRUCT_CORRIDOR, STRUCT_PALETTE, STRUCT_VOID, classid_to_rgb
from farmsim_farm.topdown_projection import load_topdown_meta


def masked_lowpass(height_grid, valid_mask, radius_px):
    """Local mean of height_grid over a (2*radius_px+1)-wide window,
    counting only valid_mask==True cells -- a plain uniform_filter would
    bias the average toward 0 near void regions (edges of the mesh
    footprint), since invalid cells there don't hold a real height."""
    size = 2 * radius_px + 1
    filled = np.where(valid_mask, height_grid, 0.0)
    weight = valid_mask.astype(np.float64)
    mean_filled = uniform_filter(filled, size=size, mode='nearest')
    mean_weight = uniform_filter(weight, size=size, mode='nearest')
    return np.where(mean_weight > 1e-6, mean_filled / mean_weight, height_grid)


def suppress_short_corridors(corridor_mask, min_length_px, row_axis='x'):
    """Keep a corridor pixel only if it's part of an unbroken run of at
    least min_length_px along the row direction, in its own raster
    row/column -- binary_opening (erode then dilate) with a long, thin
    structuring element does exactly this: a pixel survives erosion only
    if min_length_px consecutive pixels around it, along that one axis,
    are all foreground. Removes isolated dips (real terrain/mesh noise, a
    slightly-lower patch of otherwise-normal crop) that happened to cross
    the depth threshold without being part of a real, long corridor;
    leaves genuine corridors (which span meters) untouched."""
    if row_axis == 'x':
        structure = np.ones((1, min_length_px), dtype=bool)
    elif row_axis == 'y':
        structure = np.ones((min_length_px, 1), dtype=bool)
    else:
        raise ValueError(f"row_axis must be 'x' or 'y', got {row_axis!r}")
    return binary_opening(corridor_mask, structure=structure)


def classify_structure(residual, valid_mask, corridor_depth_m, min_length_px, row_axis='x'):
    """STRUCT_VOID outside the mesh footprint, STRUCT_CORRIDOR where the
    residual drops at least corridor_depth_m below the local trend AND
    that survives suppress_short_corridors, STRUCT_BED everywhere else
    in-footprint."""
    class_ids = np.full(residual.shape, STRUCT_VOID, dtype=np.uint8)
    class_ids[valid_mask] = STRUCT_BED
    raw_corridor = valid_mask & (residual < -corridor_depth_m)
    corridor = suppress_short_corridors(raw_corridor, min_length_px, row_axis)
    class_ids[corridor] = STRUCT_CORRIDOR
    return class_ids


def classify_structure_from_plots(height_tree, mesh_z, georef, resolution, topdown_meta,
                                   plots, block_corners_px):
    """Plot-polygon mode: for every cell of the output world-XY grid,
    project it (via its real terrain height, same pattern as
    build_label_raster.py) into the birdseye photo, then classify by direct
    polygon membership -- BED if inside any annotated plot, CORRIDOR if
    inside the annotated block boundary but not in a plot, VOID otherwise
    (outside the block, or the projection is invalid/off-photo). No relief
    threshold, no corridor-length heuristic -- the boundaries are exactly
    what was annotated.

    Returns (class_ids (h_px, w_px) uint8, w_px, h_px).
    """
    xs, ys, grid_x, grid_y, flat_x, flat_y, w_px, h_px = rc.build_output_grid(georef, resolution)

    flat_z = rc.lookup_heights(height_tree, mesh_z, flat_x, flat_y)
    points_w = np.column_stack([flat_x, flat_y, flat_z])
    u, v, valid = rc.world_to_pixel(points_w, topdown_meta)

    photo_shape = (topdown_meta['height'], topdown_meta['width'])
    bed_mask = rasterize_plot_footprint(plots, photo_shape)
    block_mask = rasterize_polygon(block_corners_px, photo_shape)

    # The block footprint decides what counts as CORRIDOR: corridor is
    # everything inside the block that is not a bed, i.e. the bare soil between
    # the crop ridges. If the annotated lattice does not actually enclose the
    # plots, that subtraction yields nothing and the corridor class silently
    # comes out empty -- which is exactly what happened on 20250425, where
    # corners_px holds only three nearly-collinear points (a 124k px slice
    # against a 10.7M px bed union, so 99% of the beds sat OUTSIDE their own
    # block). The result was a structure channel that was 0.07% corridor and,
    # downstream, a segmentation class that could never be learned.
    #
    # Rather than trust a degenerate annotation, derive the block from the
    # plots themselves: the convex hull of every plot polygon encloses the beds
    # and the soil strips between them by construction. The annotated lattice
    # still wins whenever it is actually valid.
    bed_area = int((bed_mask > 0).sum())
    beds_outside = int(((bed_mask > 0) & (block_mask == 0)).sum())
    if bed_area and beds_outside > 0.05 * bed_area:
        hull = cv2.convexHull(np.concatenate(
            [np.round(p['polygon_px']).astype(np.int32) for p in plots]))
        block_mask = np.zeros(photo_shape, dtype=np.uint8)
        cv2.fillPoly(block_mask, [hull], 255)
        print(f'WARNING: the annotated plot lattice does not enclose the plots '
              f'({100.0 * beds_outside / bed_area:.1f}% of bed pixels fall outside it). '
              f'Falling back to the convex hull of the plot polygons as the block '
              f'footprint, so CORRIDOR is the soil between the beds rather than empty. '
              f'Re-annotate plot_lattice corners_px to override this.')

    ui = np.clip(np.round(u).astype(int), 0, topdown_meta['width'] - 1)
    vi = np.clip(np.round(v).astype(int), 0, topdown_meta['height'] - 1)

    is_bed = np.zeros(flat_x.shape[0], dtype=bool)
    is_bed[valid] = bed_mask[vi[valid], ui[valid]] > 0
    is_in_block = np.zeros(flat_x.shape[0], dtype=bool)
    is_in_block[valid] = block_mask[vi[valid], ui[valid]] > 0

    class_ids = np.full(flat_x.shape[0], STRUCT_VOID, dtype=np.uint8)
    class_ids[is_in_block] = STRUCT_CORRIDOR
    class_ids[is_bed] = STRUCT_BED  # bed is a subset of the block by construction; wins where both hold
    return class_ids.reshape(h_px, w_px), w_px, h_px


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--georef', required=True, help='georef_*.json from bake_world_mesh.py')
    parser.add_argument('--mesh', required=True, help='label_mesh_*.glb from bake_world_mesh.py')
    parser.add_argument('--topdown-meta', default=None,
                         help='topdown_*.yaml from capture_topdown.py -- required for --plot-polygons mode')
    parser.add_argument('--plot-polygons', default=None,
                         help='plot_polygons_*.geojson from annotate_plots.py -- if given, use '
                              'plot-polygon mode instead of relief detection')
    parser.add_argument('--plot-lattice', default=None,
                         help='plot_lattice_*.json from annotate_plots.py (for the block-boundary '
                              'corners) -- required together with --plot-polygons')
    parser.add_argument('--resolution', type=float, default=0.05,
                         help="meters per output pixel -- coarser than the vegetation raster's "
                              '0.02 default is fine here, relief varies over meters not centimeters')
    parser.add_argument('--detrend-radius-m', type=float, default=10.0,
                         help='local low-pass radius for detrending, in meters -- see module '
                              "docstring for how the default was sized against this field's pitch")
    parser.add_argument('--corridor-depth-m', type=float, default=0.03,
                         help='residual (below the local trend) at which a cell counts as corridor')
    parser.add_argument('--corridor-min-length-m', type=float, default=2.0,
                         help='minimum unbroken run length along --row-axis for a corridor blob to '
                              'survive -- suppresses isolated low spots that are not real corridors')
    parser.add_argument('--row-axis', choices=['x', 'y'], default='x',
                         help="which world axis the rows run along in this survey (confirmed 'x' "
                              'for 20250425, ~0.0 deg heading from config/rows_20250425.yaml) -- an '
                              'axis-aligned opening only works when this is right; a survey with '
                              'rows at a real angle would need a rotated structuring element instead')
    parser.add_argument('--max-nn-dist-m', type=float, default=0.5,
                         help='nearest mesh-vertex distance beyond which a grid cell is treated as '
                              'outside the real mesh footprint (void), not extrapolated')
    parser.add_argument('--roi', type=int, nargs=4, default=None, metavar=('X0', 'X1', 'Y0', 'Y1'),
                         help="pixel box (in THIS raster's own w_px/h_px grid, not world meters) to "
                              'restrict output to -- e.g. to exclude a headland/border strip')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    if bool(args.plot_polygons) != bool(args.plot_lattice):
        raise SystemExit('--plot-polygons and --plot-lattice must be given together')

    with open(args.georef) as f:
        georef = json.load(f)

    print('Loading baked mesh for height lookup...')
    height_tree, mesh_z = rc.load_height_lookup(args.mesh)

    if args.plot_polygons:
        if not args.topdown_meta:
            raise SystemExit('--topdown-meta is required together with --plot-polygons')
        print('Plot-polygon mode: rasterizing structure directly from annotated boundaries...')
        topdown_meta = load_topdown_meta(args.topdown_meta)
        plots = load_plot_polygons(args.plot_polygons)
        with open(args.plot_lattice) as f:
            block_corners_px = np.array(json.load(f)['corners_px'], dtype=np.float64)

        class_ids, w_px, h_px = classify_structure_from_plots(
            height_tree, mesh_z, georef, args.resolution, topdown_meta, plots, block_corners_px)
        print(f'Output raster: {w_px} x {h_px} px at {args.resolution} m/px '
              f'({georef["width_m"]:.1f} x {georef["height_m"]:.1f} m)')
    else:
        _, _, _, _, flat_x, flat_y, w_px, h_px = rc.build_output_grid(georef, args.resolution)
        print(f'Output raster: {w_px} x {h_px} px at {args.resolution} m/px '
              f'({georef["width_m"]:.1f} x {georef["height_m"]:.1f} m)')

        print('Looking up terrain height per raster cell (nearest-vertex)...')
        flat_z, flat_dist = rc.lookup_heights(height_tree, mesh_z, flat_x, flat_y, return_dist=True)

        height_grid = flat_z.reshape(h_px, w_px)
        valid_mask = flat_dist.reshape(h_px, w_px) < args.max_nn_dist_m

        if args.roi is not None:
            x0, x1, y0, y1 = args.roi
            roi_mask = np.zeros_like(valid_mask)
            roi_mask[y0:y1, x0:x1] = True
            valid_mask &= roi_mask

        radius_px = max(1, round(args.detrend_radius_m / args.resolution))
        print(f'Detrending (radius {args.detrend_radius_m}m -> {radius_px}px window)...')
        lowpass = masked_lowpass(height_grid, valid_mask, radius_px)
        residual = height_grid - lowpass

        min_length_px = max(1, round(args.corridor_min_length_m / args.resolution))
        class_ids = classify_structure(residual, valid_mask, args.corridor_depth_m, min_length_px, args.row_axis)

    raster = classid_to_rgb(class_ids, STRUCT_PALETTE)
    Image.fromarray(raster[:, :, ::-1], 'RGB').save(args.out)  # BGR tuples -> RGB file
    print(f'Wrote {args.out}')

    total = h_px * w_px
    n_void = int((class_ids == STRUCT_VOID).sum())
    n_bed = int((class_ids == STRUCT_BED).sum())
    n_corridor = int((class_ids == STRUCT_CORRIDOR).sum())
    print(f'Class fractions: void={n_void / total:.3f} bed={n_bed / total:.3f} corridor={n_corridor / total:.3f}')

    rc.write_raster_georef(georef, args.out, args.resolution, w_px, h_px)
    print(f'Wrote raster georeference to {args.out.rsplit(".", 1)[0]}_georef.json')


if __name__ == '__main__':
    main()
