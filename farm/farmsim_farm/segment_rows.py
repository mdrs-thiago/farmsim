"""
segment_rows.py

Automatic alternative to mark_rows.py's manual point-clicking: segments the
top-down capture into "plantation" (crop foliage) vs "row" (the bare-soil
corridor between crop strips, i.e. where the robot's wheels can actually
drive), then extracts one drivable centerline per corridor automatically.
Writes the same rows.yaml schema mark_rows.py does, so it's a drop-in
replacement usable as-is by row_geometry.py / row_follower.py -- plus an
overlay PNG so you can eyeball the result instead of having to click
anything.

Method:
  1. HSV-hue threshold (see vegetation_mask) -> binary plantation mask.
     Deliberately hue, not a green-index like ExG: this photogrammetry
     render has a brightness/exposure gradient across the field (texture
     atlas seam), which defeats any global magnitude-based threshold; hue
     is brightness-invariant and separates crop from soil/path/headland
     cleanly everywhere. Requires the capture to be reasonably
     high-resolution/low-altitude -- at the original 1600px/65m capture
     the inter-row corridors were only ~2-4px wide, too thin for either
     this pipeline or a human eye to place reliably; recaptured at
     3200px/50m (~3.1cm/px) instead.
  2. Fit a rotated bounding rect to the plantation mask (cv2.minAreaRect)
     and warp so crop rows run vertical (extract_row_corridors).
  3. This field is a dense grid of research plots, not simple parallel
     rows: most gaps in the vegetation mask are short perpendicular
     plot-to-plot boundaries, not through-going lanes. Open with a tall,
     1px-wide vertical kernel -- erosion with it only survives a strict
     vertical run of corridor pixels, which the short horizontal
     cross-paths never satisfy, so they're erased and only long vertical
     corridors remain as separate connected components.
  4. Keep components spanning a large fraction of the field's length,
     sample a centerline per component (mean x within a small y-band,
     tolerating slight curvature), map back through the inverse warp and
     mark_rows.pixel_to_world's ray-cast to world (x, y).
  5. merge_collinear_rows fuses corridor fragments that are actually the
     same physical lane split by a plot-boundary crossing (same heading,
     same lateral offset, endpoints close together) into one row.

Usage:
  python3 segment_rows.py --meta topdown_20250624.yaml \
      --output rows_20250624.yaml --overlay overlay_20250624.png
"""

import argparse

import cv2
import numpy as np
import yaml

from farmsim_farm.topdown_projection import pixel_to_world
from farmsim_farm.topdown_projection import load_topdown_meta


def vegetation_mask(image_bgr, hue_min=36, sat_min=40, close_kernel_height=0):
    """HSV-hue threshold, not a green-channel index (ExG etc): this
    photogrammetry render has a visible brightness/exposure gradient
    across the field (likely a texture-atlas seam -- the mesh is stitched
    from 12 separate atlases), which defeats any single global threshold
    on a magnitude-based index (confirmed empirically: ExG+Otsu cleanly
    separated crop/soil on the image's left third and missed it entirely
    on the right, tracking that gradient). Hue is invariant to
    brightness, and sampling actual pixels here showed a clean, uniform
    split -- crop foliage sits at hue~40-50 (0-179 OpenCV scale) with
    sat~120-180, soil/path/headland sits at hue~20-33 with sat~50-100,
    consistently across the whole image.

    close_kernel_height (0 = disabled, matches the original 20250624
    behavior exactly) is for early-growth-stage surveys: a full canopy
    (20250624, thresholds tuned above) is one solid hue/sat blob per plot,
    so no closing is needed or wanted -- the drivable corridors are only a
    few pixels wide at that resolution, and closing would erase them along
    with genuine noise. But a sparse, still-emerging crop (confirmed on
    the 20250425 survey: individual drill-seeded row lines with visible
    bare soil between them, not a solid canopy -- mean hue sits right at
    the same threshold that cleanly separates full canopy, so the raw mask
    is a fine, gappy grid rather than solid blocks) needs those
    within-drill-row gaps bridged before the corridor-extraction step
    below has solid blocks to work with. A TALL, ONE-PIXEL-WIDE vertical
    kernel closes gaps along the row direction only (same shape as
    extract_row_corridors' own vertical-open kernel, just closing instead
    of opening, and applied earlier) -- it can't bridge sideways into a
    neighboring drill row or across an actual corridor, so it's safe to
    leave on for any survey; it's just a no-op when the canopy is already
    solid (nothing left to close)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    mask = ((h > hue_min) & (s > sat_min)).astype(np.uint8) * 255
    if close_kernel_height > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_kernel_height))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    # Deliberately no isotropic MORPH_CLOSE here: see close_kernel_height
    # above. Just a light open to remove single-pixel speckle.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def field_rotated_rect(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('No plantation contour found -- vegetation mask is empty')
    largest = max(contours, key=cv2.contourArea)
    return cv2.minAreaRect(largest)


def warp_to_axis_aligned(mask, rect):
    """Rotate so the field's long axis (crop row direction) is vertical.
    Returns the full-size warped mask, the rotation matrix used, and the
    field's axis-aligned bounding box within the warped frame."""
    (cx, cy), (w, h), angle = rect
    if w > h:
        angle += 90.0
        w, h = h, w
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    warped = cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]))
    x0 = int(round(cx - w / 2.0))
    x1 = int(round(cx + w / 2.0))
    y0 = int(round(cy - h / 2.0))
    y1 = int(round(cy + h / 2.0))
    return warped, m, (x0, y0, x1, y1)


def extract_row_corridors(veg_mask, rect, min_length_frac=0.4):
    """The path network is a grid (long row-direction corridors crossed
    periodically by short perpendicular access paths between plot
    blocks), so a plain connected-components pass treats the whole grid
    as one blob -- every row corridor touches its cross-paths, and often
    the field's own headland border too. Fix: warp so rows run vertical,
    then morphologically open with a *tall, one-pixel-wide vertical*
    kernel. Erosion with that kernel only survives where a full
    kernel-length run of corridor pixels exists in a strictly vertical
    line, which the short horizontal cross-paths never satisfy -- so they
    (and the headland) get erased, leaving the long vertical row
    corridors as separate connected components.

    Returns a list of (ys_warped, xs_warped) pixel-coordinate arrays (one
    per accepted corridor, in the WARPED frame), the rotation matrix, and
    the field's axis-aligned bbox in the warped frame."""
    warped_veg, m, (x0, y0, x1, y1) = warp_to_axis_aligned(veg_mask, rect)
    field_h = y1 - y0

    corridor = cv2.bitwise_not(warped_veg)
    bbox_mask = np.zeros_like(corridor)
    bbox_mask[y0:y1, x0:x1] = 255
    corridor = cv2.bitwise_and(corridor, bbox_mask)

    k_len = max(15, int(0.06 * field_h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(corridor, connectivity=8)

    corridors = []
    for label in range(1, num):
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if height < min_length_frac * field_h:
            continue
        ys, xs = np.where(labels == label)
        corridors.append((ys, xs))

    corridors.sort(key=lambda c: c[1].mean())
    return corridors, m, (x0, y0, x1, y1)


def build_rows(mask, meta, rect, n_samples=8, inset_frac=0.05, min_length_frac=0.4):
    corridors, m, (x0, y0, x1, y1) = extract_row_corridors(mask, rect, min_length_frac=min_length_frac)
    m_inv = cv2.invertAffineTransform(m)

    rows_px = []
    rows_world = []
    for ys, xs in corridors:
        v_lo = ys.min() + inset_frac * (ys.max() - ys.min())
        v_hi = ys.max() - inset_frac * (ys.max() - ys.min())
        v_samples = np.linspace(v_lo, v_hi, n_samples)

        pts_orig = []
        for v in v_samples:
            # Centerline x at this height: mean x of corridor pixels
            # within a small band around v (tolerates slight curvature).
            band = np.abs(ys - v) < 3.0
            if not np.any(band):
                nearest = np.argmin(np.abs(ys - v))
                u_warped = float(xs[nearest])
            else:
                u_warped = float(xs[band].mean())
            pt = np.array([[u_warped, v]], dtype=np.float32).reshape(1, 1, 2)
            orig = cv2.transform(pt, m_inv).reshape(2)
            pts_orig.append((float(orig[0]), float(orig[1])))
        rows_px.append(pts_orig)

        world_pts = []
        for (u, v) in pts_orig:
            x, y = pixel_to_world(u, v, meta)
            world_pts.append([round(x, 3), round(y, 3)])
        rows_world.append(world_pts)

    return merge_collinear_rows(rows_px, rows_world)


def merge_collinear_rows(rows_px, rows_world, angle_tol_deg=8.0, perp_tol=1.2, gap_tol=6.0):
    """The perpendicular block-boundary paths that separate corridors into
    connected components (see extract_row_corridors) sometimes cut clean
    through what is, physically, a single straight lane -- two components
    at (almost) the same lateral offset and heading, with their nearer
    ends close together. Union-find over that similarity, merging any
    such pair (and transitively, chains of more than two) into one row so
    row_geometry/row_follower see one continuous lane instead of two
    coincidentally-adjacent ones."""
    world_arrs = [np.array(pts, dtype=np.float64) for pts in rows_world]
    n = len(world_arrs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    def line_params(pts):
        d = pts[-1] - pts[0]
        norm = np.linalg.norm(d)
        direction = d / norm if norm > 1e-6 else np.array([1.0, 0.0])
        return pts[0], direction

    for i in range(n):
        pi0, di = line_params(world_arrs[i])
        for j in range(i + 1, n):
            pj0, dj = line_params(world_arrs[j])
            if abs(np.dot(di, dj)) < np.cos(np.deg2rad(angle_tol_deg)):
                continue
            perp = np.array([-di[1], di[0]])
            if abs(np.dot(pj0 - pi0, perp)) > perp_tol:
                continue
            ends_i = (world_arrs[i][0], world_arrs[i][-1])
            ends_j = (world_arrs[j][0], world_arrs[j][-1])
            min_gap = min(np.linalg.norm(a - b) for a in ends_i for b in ends_j)
            if min_gap > gap_tol:
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged_px, merged_world = [], []
    for members in sorted(groups.values(), key=lambda ms: world_arrs[ms[0]][0, 0]):
        all_world = np.concatenate([world_arrs[m] for m in members], axis=0)
        all_px = np.concatenate([np.array(rows_px[m]) for m in members], axis=0)
        _, direction = line_params(world_arrs[members[0]])
        proj = (all_world - all_world[0]) @ direction
        order = np.argsort(proj)
        merged_world.append([[round(float(x), 3), round(float(y), 3)] for x, y in all_world[order]])
        merged_px.append([(float(u), float(v)) for u, v in all_px[order]])

    return merged_px, merged_world


def draw_overlay(image_bgr, mask, rows_px, rect):
    overlay = image_bgr.copy()
    green_tint = np.zeros_like(overlay)
    green_tint[:, :, 1] = 255
    veg = mask > 0
    overlay[veg] = cv2.addWeighted(overlay, 0.6, green_tint, 0.4, 0)[veg]

    box = cv2.boxPoints(rect).astype(int)
    cv2.polylines(overlay, [box], True, (255, 255, 0), 3)

    colors = [(0, 0, 255), (255, 0, 255), (0, 165, 255), (255, 0, 0),
              (0, 255, 255), (128, 0, 255), (0, 128, 255), (255, 128, 0)]
    for i, pts in enumerate(rows_px):
        color = colors[i % len(colors)]
        pts_int = [(int(round(u)), int(round(v))) for (u, v) in pts]
        for a, b in zip(pts_int, pts_int[1:]):
            cv2.line(overlay, a, b, color, 4)
        for p in pts_int:
            cv2.circle(overlay, p, 6, color, -1)
        cv2.putText(overlay, str(i), pts_int[0], cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    color, 3, cv2.LINE_AA)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta', required=True, help='Path to topdown_*.yaml from capture_topdown')
    parser.add_argument('--output', required=True, help='Path to write rows_*.yaml')
    parser.add_argument('--overlay', required=True, help='Path to write a sanity-check overlay PNG')
    parser.add_argument('--hue-min', type=int, default=36,
                        help='OpenCV hue (0-179) above which a pixel counts as crop foliage')
    parser.add_argument('--sat-min', type=int, default=40,
                        help='Saturation floor to reject the grey background outside the field')
    parser.add_argument('--min-length-frac', type=float, default=0.4,
                        help='Minimum corridor length as a fraction of the field long dimension')
    parser.add_argument('--close-kernel-height', type=int, default=0,
                        help='Vertical closing kernel height in px, for early-growth-stage '
                             "surveys where the raw vegetation mask is a sparse, gappy grid "
                             "rather than solid per-plot blocks (0 = disabled, matches the "
                             "original full-canopy behavior).")
    parser.add_argument('--roi', type=int, nargs=4, default=None,
                        metavar=('X0', 'X1', 'Y0', 'Y1'),
                        help='Pixel box to restrict segmentation to, e.g. --roi 1412 4987 193 '
                             '6228. Needed when the hue/sat thresholds loose enough to catch a '
                             "sparse/early-growth crop also catch vegetation outside the actual "
                             "plot grid (e.g. a grassy headland or border strip) -- that "
                             'contamination otherwise throws off field_rotated_rect (the field '
                             'boundary it fits balloons out to include the border) and, '
                             'downstream, the min-length-frac filter (now measured against that '
                             'inflated boundary, so real corridors no longer look long enough). '
                             'Not needed for a clean full-canopy mask like 20250624\'s.')
    args = parser.parse_args()

    meta = load_topdown_meta(args.meta)

    image = cv2.imread(meta['image_path'])
    if image is None:
        raise FileNotFoundError(f"Could not load image: {meta['image_path']}")

    mask = vegetation_mask(image, hue_min=args.hue_min, sat_min=args.sat_min,
                            close_kernel_height=args.close_kernel_height)
    if args.roi is not None:
        x0, x1, y0, y1 = args.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = roi_mask
    rect = field_rotated_rect(mask)
    rows_px, rows_world = build_rows(mask, meta, rect, min_length_frac=args.min_length_frac)

    data = {'rows': [{'id': i, 'points': pts} for i, pts in enumerate(rows_world)]}
    with open(args.output, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f'Wrote {len(rows_world)} row(s) to {args.output}')

    overlay = draw_overlay(image, mask, rows_px, rect)
    cv2.imwrite(args.overlay, overlay)
    print(f'Wrote overlay to {args.overlay}')


if __name__ == '__main__':
    main()
