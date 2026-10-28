"""
plot_lattice.py

Pure math for the parametric plot-grid annotation tool (annotate_plots.py):
fit a homography from 4 clicked block corners, generate a regular n_rows x
n_cols grid of alley-inset plot rectangles in the homography's normalized
[0,1]x[0,1] space, optionally nudge individual rows/columns to absorb gentle
real-world misalignment/curvature, then map everything to pixel coordinates
for drawing/export. No cv2/GUI dependency -- fully unit-testable, and reused
as-is by exg_classify.py (Phase B) via the exported plot polygons, not
reimplemented there.

Normalized-space convention: x in [0,1] is the "column" (n_cols) direction,
y in [0,1] is the "row" (n_rows) direction, y increasing the same way v
(image row) does -- i.e. corners must be clicked/passed in image order
[top-left, top-right, bottom-right, bottom-left].
"""

import numpy as np

# Normalized-space corners the homography maps FROM, in the same
# [top-left, top-right, bottom-right, bottom-left] order callers must supply
# pixel corners in.
_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def fit_block_homography(corner_points_px):
    """corner_points_px: 4 (u, v) pixel points, in order
    [top-left, top-right, bottom-right, bottom-left].

    Returns (H, H_inv): H maps normalized [0,1]x[0,1] block coordinates to
    pixel coordinates; H_inv is its exact algebraic inverse (pixel ->
    normalized). Solved via direct linear transform (DLT) with h33 fixed to
    1 -- exact for 4 correspondences, no least-squares approximation."""
    corners = np.asarray(corner_points_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f'Expected exactly 4 (u, v) corner points, got shape {corners.shape}')

    rows = []
    rhs = []
    for (x, y), (u, v) in zip(_UNIT_SQUARE, corners):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)

    h = np.linalg.solve(np.array(rows), np.array(rhs))
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]])
    H_inv = np.linalg.inv(H)
    return H, H_inv


def apply_homography(H, points):
    """points: (N, 2) array of (x, y). Returns (N, 2) transformed points,
    with the homogeneous divide applied."""
    points = np.asarray(points, dtype=np.float64)
    homog = np.column_stack([points, np.ones(len(points))])
    mapped = homog @ H.T
    return mapped[:, :2] / mapped[:, 2:3]


def generate_plot_grid(n_rows, n_cols, alley_width_fraction):
    """Regular n_rows x n_cols grid of alley-inset plot rectangles in
    normalized [0,1]x[0,1] block space. alley_width_fraction is the alley's
    share of one plot's pitch (0 = plots touch edge-to-edge, must be < 1).

    Returns a list of dicts, row-major order:
      {'row': i, 'col': j,
       'corners': (4, 2) ndarray, [top-left, top-right, bottom-right, bottom-left],
       'excluded': False}
    """
    if not (0.0 <= alley_width_fraction < 1.0):
        raise ValueError(f'alley_width_fraction must be in [0, 1), got {alley_width_fraction}')
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f'n_rows/n_cols must be >= 1, got {n_rows}/{n_cols}')

    pitch_x = 1.0 / n_cols
    pitch_y = 1.0 / n_rows
    inset_x = alley_width_fraction * pitch_x / 2.0
    inset_y = alley_width_fraction * pitch_y / 2.0

    plots = []
    for i in range(n_rows):
        y0 = i * pitch_y + inset_y
        y1 = (i + 1) * pitch_y - inset_y
        for j in range(n_cols):
            x0 = j * pitch_x + inset_x
            x1 = (j + 1) * pitch_x - inset_x
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            plots.append({'row': i, 'col': j, 'corners': corners, 'excluded': False})
    return plots


def apply_nudges(plots, row_nudges=None, col_nudges=None):
    """Shift each plot's corners by that plot's row nudge (dy, added to y)
    plus its column nudge (dx, added to x), in normalized block space --
    absorbs gentle real-world misalignment/curvature the rigid homography
    fit can't, without re-clicking corners. row_nudges/col_nudges: dict of
    {index: delta}, missing entries default to 0. Returns a new list; does
    not mutate the input."""
    row_nudges = row_nudges or {}
    col_nudges = col_nudges or {}
    out = []
    for plot in plots:
        dx = col_nudges.get(plot['col'], 0.0)
        dy = row_nudges.get(plot['row'], 0.0)
        nudged = dict(plot)
        nudged['corners'] = plot['corners'] + np.array([dx, dy])
        out.append(nudged)
    return out


def apply_global_offset_scale(plots, offset=(0.0, 0.0), scale=1.0):
    """Scale the whole grid about the block center (0.5, 0.5), then
    translate by `offset`, in normalized block space -- for coarse
    whole-grid adjustment before per-row/per-column nudging. Returns a new
    list; does not mutate the input."""
    center = np.array([0.5, 0.5])
    offset = np.array(offset, dtype=np.float64)
    out = []
    for plot in plots:
        transformed = center + (plot['corners'] - center) * scale + offset
        nudged = dict(plot)
        nudged['corners'] = transformed
        out.append(nudged)
    return out


def plots_to_pixel(plots_norm, H):
    """Map each plot's normalized corners to pixel coordinates via H
    (normalized -> pixel, from fit_block_homography). Returns a new list of
    dicts with 'corners_px' added (original 'corners' left untouched, so
    callers keep both spaces without re-deriving one from the other)."""
    out = []
    for plot in plots_norm:
        out_plot = dict(plot)
        out_plot['corners_px'] = apply_homography(H, plot['corners'])
        out.append(out_plot)
    return out


def plots_to_geojson(plots_px):
    """plots_px: output of plots_to_pixel (each dict has 'corners_px').
    Returns a GeoJSON-like FeatureCollection dict, polygon coordinates in
    image-pixel space, ring closed (first point repeated last, per GeoJSON
    convention)."""
    features = []
    for idx, plot in enumerate(plots_px):
        ring = [list(map(float, p)) for p in plot['corners_px']]
        ring.append(ring[0])
        features.append({
            'type': 'Feature',
            'properties': {'index': idx, 'row': int(plot['row']), 'col': int(plot['col']),
                            'excluded': bool(plot['excluded'])},
            'geometry': {'type': 'Polygon', 'coordinates': [ring]},
        })
    return {'type': 'FeatureCollection', 'features': features}
