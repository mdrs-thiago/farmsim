"""
exg_classify.py

Vegetation classification seeded within each annotated plot polygon
(annotate_plots.py's plot_polygons_*.geojson), replacing
segment_rows.vegetation_mask's whole-image HSV-hue threshold as
build_label_raster.py's classifier.

Why this fixes what HSV was worked around: segment_rows.py's own docstring
documents that ExG+Otsu was tried first and abandoned globally -- this
photogrammetry render has a real brightness/exposure gradient across the
field (a texture-atlas seam), which defeats any single GLOBAL threshold on a
magnitude-based index like ExG ("cleanly separated crop/soil on the image's
left third and missed it entirely on the right"). Otsu-thresholding
LOCALLY, per plot polygon, sidesteps that problem by construction: each
plot spans a few meters, far smaller than the atlas-seam gradient's scale,
so within any one plot the exposure is effectively uniform and a single
threshold is valid there -- even though no single threshold would be valid
across the whole photo. This is also strictly more precise than the
row-level HSV approach for THIS field, which is a dense grid of small
plots (see segment_rows.py's own notes): a per-plot threshold responds to
that plot's own patchiness instead of one setting for an entire corridor.

Method: ExG = 2G - R - B (vectorized, whole photo, cheap), then per plot
polygon: crop to the polygon's bounding box, Otsu-threshold ONLY the pixels
inside the polygon (not the box's corners, which may include a neighboring
alley), light morphological open+close within that same crop. A plot
flagged excluded/bare in the annotation tool skips classification entirely
and is forced not-vegetated -- human-observed ground truth overrides the
automatic threshold, never the other way around.
"""

import argparse
import json

import cv2
import numpy as np
from PIL import Image
from farmsim_farm.topdown_projection import load_topdown_meta


def exg_index(bgr_image):
    """2G - R - B, vectorized over an (H, W, 3) BGR uint8 image. Returns a
    float32 (H, W) array (unbounded, not clipped/normalized -- callers
    threshold it directly, no fixed 0-255 range is assumed)."""
    b = bgr_image[..., 0].astype(np.float32)
    g = bgr_image[..., 1].astype(np.float32)
    r = bgr_image[..., 2].astype(np.float32)
    return 2.0 * g - r - b


def otsu_threshold(values):
    """Otsu's method (maximize inter-class variance) over an arbitrary 1D
    array of real-valued samples, via a 256-bin histogram spanning the
    sample's own min/max -- not cv2.threshold's THRESH_OTSU, which requires
    a full 2D uint8 image and can't be restricted to an arbitrary (masked,
    non-rectangular) set of pixel values. Degenerates to returning `vmin`
    (everything one class) when the plot is uniform (vmax <= vmin), e.g. a
    fully void or fully flat-colored crop."""
    values = np.asarray(values, dtype=np.float64)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin

    hist, bin_edges = np.histogram(values, bins=256, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = weight1[-1] - weight1
    sum1 = np.cumsum(hist * centers)
    total_sum = sum1[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = np.where(weight1 > 0, sum1 / weight1, 0.0)
        mean2 = np.where(weight2 > 0, (total_sum - sum1) / weight2, 0.0)
    between_class_var = weight1 * weight2 * (mean1 - mean2) ** 2

    # Last bin has weight2 == 0 (nothing left above it) by construction --
    # never a valid split point, exclude it before argmax.
    between_class_var[-1] = -1.0
    return float(centers[int(np.argmax(between_class_var))])


def load_plot_polygons(geojson_path):
    """Reads annotate_plots.py's plot_polygons_*.geojson. Returns a list of
    dicts: {'index', 'row', 'col', 'excluded', 'polygon_px' ((N,2) float
    ndarray, ring NOT closed -- the repeated closing point is dropped)}."""
    with open(geojson_path) as f:
        data = json.load(f)
    plots = []
    for feat in data['features']:
        ring = np.array(feat['geometry']['coordinates'][0], dtype=np.float64)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        props = feat['properties']
        plots.append({'index': props['index'], 'row': props['row'], 'col': props['col'],
                       'excluded': bool(props['excluded']), 'polygon_px': ring})
    return plots


def rasterize_polygon(polygon_px, shape):
    """polygon_px: (N, 2) ring (not necessarily closed). shape: (H, W).
    Returns an (H, W) uint8 mask, 255 inside the polygon. Shared by
    build_structure_raster.py's plot-polygon mode (bed = union of plot
    polygons, block = the single block-boundary polygon) -- not
    reimplemented there."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon_px).astype(np.int32)], 255)
    return mask


def rasterize_plot_footprint(plots, shape):
    """Union of every (non-excluded and excluded alike -- exclusion only
    affects vegetation classification, not the physical bed footprint) plot
    polygon's interior, rasterized onto an (H, W) uint8 mask, 255 = inside
    some plot. This is the BED footprint build_structure_raster.py's
    plot-polygon mode needs; classify_within_plots computes the same
    per-plot union as a side effect (its `captured_mask` return value) but
    also always runs ExG, which that caller doesn't want."""
    mask = np.zeros(shape, dtype=np.uint8)
    for plot in plots:
        cv2.fillPoly(mask, [np.round(plot['polygon_px']).astype(np.int32)], 255)
    return mask


def classify_within_plots(photo_bgr, plots, morph_kernel_size=3):
    """Per-plot local ExG+Otsu classification.

    Returns (veg_mask, captured_mask, plot_stats):
      veg_mask, captured_mask: (H, W) uint8, 0 or 255 -- veg_mask is a
        subset of captured_mask by construction (excluded plots contribute
        to captured_mask but never to veg_mask).
      plot_stats: list of dicts (one per plot, geojson order), each with
        'index', 'row', 'col', 'excluded', 'n_total_px', 'n_veg_px',
        'coverage' (n_veg_px / n_total_px, or None if the polygon rasterized
        to zero pixels -- degenerate/off-image).
    """
    h, w = photo_bgr.shape[:2]
    exg = exg_index(photo_bgr)
    veg_mask = np.zeros((h, w), dtype=np.uint8)
    captured_mask = np.zeros((h, w), dtype=np.uint8)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8) if morph_kernel_size > 0 else None

    plot_stats = []
    for plot in plots:
        ring = plot['polygon_px']
        x0 = max(0, int(np.floor(ring[:, 0].min())))
        y0 = max(0, int(np.floor(ring[:, 1].min())))
        x1 = min(w, int(np.ceil(ring[:, 0].max())))
        y1 = min(h, int(np.ceil(ring[:, 1].max())))

        stat = {'index': plot['index'], 'row': plot['row'], 'col': plot['col'],
                'excluded': plot['excluded'], 'n_total_px': 0, 'n_veg_px': 0, 'coverage': None}
        if x1 <= x0 or y1 <= y0:
            plot_stats.append(stat)
            continue

        local_poly = (ring - [x0, y0]).astype(np.int32)
        local_area = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(local_area, [local_poly], 255)
        area_bool = local_area > 0
        n_total = int(area_bool.sum())
        stat['n_total_px'] = n_total
        if n_total == 0:
            plot_stats.append(stat)
            continue

        captured_mask[y0:y1, x0:x1][area_bool] = 255

        if plot['excluded']:
            local_veg = np.zeros_like(area_bool)
        else:
            local_exg = exg[y0:y1, x0:x1]
            threshold = otsu_threshold(local_exg[area_bool])
            local_veg = area_bool & (local_exg > threshold)
            if kernel is not None:
                veg_u8 = (local_veg * 255).astype(np.uint8)
                veg_u8 = cv2.morphologyEx(veg_u8, cv2.MORPH_OPEN, kernel)
                veg_u8 = cv2.morphologyEx(veg_u8, cv2.MORPH_CLOSE, kernel)
                # Re-clip to the polygon: MORPH_CLOSE can dilate a couple
                # pixels past the true boundary near the edge.
                local_veg = (veg_u8 > 0) & area_bool

        veg_mask[y0:y1, x0:x1][local_veg] = 255
        n_veg = int(local_veg.sum())
        stat['n_veg_px'] = n_veg
        stat['coverage'] = n_veg / n_total
        plot_stats.append(stat)

    return veg_mask, captured_mask, plot_stats


def plot_coverage_report(plot_stats, outlier_threshold=0.3):
    """Per-plot coverage summary + outlier flag vs. the field-wide median
    (computed over non-excluded plots with a valid coverage value only --
    an excluded plot's forced-zero coverage isn't a real measurement and
    would just drag the median down). Flags a plot when its coverage
    differs from that median by more than `outlier_threshold` (absolute
    fraction) -- deliberately simple/inspectable rather than a statistical
    model, since this report's purpose is surfacing candidates for human
    review, not making the final call."""
    covered = [s['coverage'] for s in plot_stats if not s['excluded'] and s['coverage'] is not None]
    median = float(np.median(covered)) if covered else None

    report = []
    for s in plot_stats:
        outlier = (median is not None and not s['excluded'] and s['coverage'] is not None
                   and abs(s['coverage'] - median) > outlier_threshold)
        report.append({**s, 'field_median_coverage': median, 'outlier': bool(outlier)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topdown-meta', required=True, help='topdown_*.yaml from capture_topdown.py')
    parser.add_argument('--plot-polygons', required=True, help='plot_polygons_*.geojson from annotate_plots.py')
    parser.add_argument('--morph-kernel-size', type=int, default=3)
    parser.add_argument('--outlier-threshold', type=float, default=0.3)
    parser.add_argument('--out-veg-mask', default=None, help='Optional: write the binary veg mask PNG')
    parser.add_argument('--out-coverage-report', required=True, help='Path to write coverage_report_*.json')
    args = parser.parse_args()

    import yaml
    topdown_meta = load_topdown_meta(args.topdown_meta)
    photo = np.asarray(Image.open(topdown_meta['image_path']).convert('RGB'))[:, :, ::-1]  # RGB -> BGR

    plots = load_plot_polygons(args.plot_polygons)
    veg_mask, captured_mask, plot_stats = classify_within_plots(photo, plots, args.morph_kernel_size)

    report = plot_coverage_report(plot_stats, args.outlier_threshold)
    with open(args.out_coverage_report, 'w') as f:
        json.dump(report, f, indent=2)
    n_outliers = sum(1 for r in report if r['outlier'])
    print(f'Wrote coverage report for {len(report)} plot(s) to {args.out_coverage_report} '
          f'({n_outliers} flagged outlier(s))')

    if args.out_veg_mask:
        Image.fromarray(veg_mask).save(args.out_veg_mask)
        print(f'Wrote veg mask to {args.out_veg_mask}')


if __name__ == '__main__':
    main()
