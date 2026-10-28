"""
build_label_raster.py

Phase 3 of the mesh-based label-rendering pipeline: turn a birdseye-photo
segmentation into a PNG whose pixel grid is EXACT world XY, at a fixed
resolution, extent matching the Phase 2 georeference exactly.

Classification source: exg_classify.classify_within_plots, seeded within
each polygon of annotate_plots.py's plot_polygons_*.geojson (ExG+Otsu,
thresholded LOCALLY per plot -- see that module's docstring for why this
supersedes segment_rows.vegetation_mask's whole-image HSV threshold: a
per-plot threshold isn't defeated by the field-wide brightness gradient a
single global threshold is).

This is now the VEGETATION channel only (vegetated / not-vegetated /
void-outside-capture) -- row/corridor structure is a separate, purely
geometric signal now (see build_structure_raster.py), so this script no
longer frames its non-crop class as "road/drivable"; it's just "captured
but not showing vegetation."

Why this step exists, not just reusing the birdseye photo/mask directly:
the birdseye photo is a real PERSPECTIVE capture, not an orthophoto -- its
pixel grid only matches world XY exactly on a perfectly flat plane at the
one assumed ground_plane_z. This terrain has real relief (confirmed:
~0.2m even along one fairly flat row, more elsewhere), so treating the
photo's pixel grid as if it were already an orthographic world-XY grid
silently introduces a position error that grows with a point's height
above/below that one assumed plane -- the same class of error already
found and fixed once for the front-camera projection (ground_plane_z),
now happening again one step upstream, at RASTER-BUILD time instead of
render time.

Fix: for every cell of the OUTPUT world-XY grid, look up that cell's real
terrain height from the baked mesh (nearest-vertex, via the Phase 1/2
mesh -- dense enough here, ~18cm average vertex spacing, to noticeably
improve on a single global constant without needing full ray-mesh
intersection), then project THAT true 3D point into the birdseye photo
to sample its class. This is the forward/inverse of
topdown_projection.pixel_to_world's ray-cast -- same camera model,
opposite direction, real height instead of an assumed flat one.
(Grid-building, height-lookup, and projection are shared with
build_structure_raster.py/check_alignment.py via raster_common.py -- not
reimplemented here.)

Usage:
  python farm/tools/build_label_raster.py \
      --georef farm/fields/20250425/georef.json \
      --mesh farm/fields/20250425/label_mesh.glb \
      --topdown-meta farm/fields/20250425/topdown.yaml \
      --plot-polygons config/plot_polygons_20250425.geojson \
      --resolution 0.02 \
      --out farm/fields/20250425/vegetation_raster.png
"""

import argparse
import json

import numpy as np
import yaml
from PIL import Image

from farmsim_farm import raster_common as rc
from exg_classify import classify_within_plots, load_plot_polygons, plot_coverage_report
from farmsim_farm.palettes import VEG_PALETTE, classid_to_rgb
from farmsim_farm.topdown_projection import load_topdown_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--georef', required=True, help='georef_*.json from bake_world_mesh.py')
    parser.add_argument('--mesh', required=True, help='label_mesh_*.glb from bake_world_mesh.py')
    parser.add_argument('--topdown-meta', required=True, help='topdown_*.yaml from capture_topdown.py')
    parser.add_argument('--plot-polygons', required=True, help='plot_polygons_*.geojson from annotate_plots.py')
    parser.add_argument('--resolution', type=float, default=0.02, help='meters per output pixel')
    parser.add_argument('--morph-kernel-size', type=int, default=3)
    parser.add_argument('--outlier-threshold', type=float, default=0.3,
                         help='flag a plot in the coverage report if its coverage differs from the '
                              'field median by more than this (absolute fraction)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.georef) as f:
        georef = json.load(f)
    topdown_meta = load_topdown_meta(args.topdown_meta)

    photo = np.asarray(Image.open(topdown_meta['image_path']).convert('RGB'))[:, :, ::-1]  # RGB -> BGR
    plots = load_plot_polygons(args.plot_polygons)
    veg_mask, captured_mask, plot_stats = classify_within_plots(photo, plots, args.morph_kernel_size)

    report = plot_coverage_report(plot_stats, args.outlier_threshold)
    coverage_report_path = args.out.rsplit('.', 1)[0] + '_coverage_report.json'
    with open(coverage_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    n_outliers = sum(1 for r in report if r['outlier'])
    print(f'Wrote coverage report for {len(report)} plot(s) to {coverage_report_path} '
          f'({n_outliers} flagged outlier(s) for review)')

    print('Loading baked mesh for height lookup...')
    height_tree, mesh_z = rc.load_height_lookup(args.mesh)

    xs, ys, grid_x, grid_y, flat_x, flat_y, w_px, h_px = rc.build_output_grid(georef, args.resolution)
    print(f'Output raster: {w_px} x {h_px} px at {args.resolution} m/px '
          f'({georef["width_m"]:.1f} x {georef["height_m"]:.1f} m)')

    print('Looking up terrain height per raster cell (nearest-vertex)...')
    flat_z = rc.lookup_heights(height_tree, mesh_z, flat_x, flat_y)

    print('Projecting raster cells into the birdseye photo...')
    points_w = np.column_stack([flat_x, flat_y, flat_z])
    u, v, valid = rc.world_to_pixel(points_w, topdown_meta)

    ui = np.clip(np.round(u).astype(int), 0, topdown_meta['width'] - 1)
    vi = np.clip(np.round(v).astype(int), 0, topdown_meta['height'] - 1)

    is_veg = np.zeros(flat_x.shape[0], dtype=bool)
    is_veg[valid] = veg_mask[vi[valid], ui[valid]] > 0
    is_captured = np.zeros(flat_x.shape[0], dtype=bool)
    is_captured[valid] = captured_mask[vi[valid], ui[valid]] > 0
    # is_veg is already a subset of is_captured by construction
    # (classify_within_plots's veg_mask is a subset of its captured_mask
    # -- every vegetated pixel is inside some plot polygon), so no
    # separate widening is needed here.

    class_ids = np.zeros(flat_x.shape[0], dtype=np.uint8)  # VEG_VOID
    class_ids[is_captured] = 1  # VEG_NOT_VEG
    class_ids[is_veg] = 2       # VEG_VEG
    class_ids = class_ids.reshape(h_px, w_px)

    raster = classid_to_rgb(class_ids, VEG_PALETTE)
    Image.fromarray(raster[:, :, ::-1], 'RGB').save(args.out)  # BGR tuples -> RGB file
    print(f'Wrote {args.out}')

    total = h_px * w_px
    n_void = int((class_ids == 0).sum())
    n_bare = int((class_ids == 1).sum())
    n_veg = int((class_ids == 2).sum())
    print(f'Class fractions: void={n_void/total:.3f} bare={n_bare/total:.3f} veg={n_veg/total:.3f}')

    rc.write_raster_georef(georef, args.out, args.resolution, w_px, h_px)
    print(f'Wrote raster georeference to {args.out.rsplit(".", 1)[0]}_georef.json')


if __name__ == '__main__':
    main()
