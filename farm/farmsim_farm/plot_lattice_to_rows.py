"""
plot_lattice_to_rows.py

Bridges annotate_plots.py's manual per-row annotation (manual_polygons_px
in plot_lattice_*.json -- each a closed, clicked quad outlining one real
crop row on the top-down photo) into the rows.yaml format row_follower.py
/ row_geometry.py already consume: converts each polygon's pixel corners
to world (x, y) via topdown_projection.pixel_to_world (the same shared
transform mark_rows.py and annotate_plots.py already use), and writes
mark_rows.py's exact output schema (closed: true), so the existing,
unmodified polygon_to_centerline.py can turn the result into a clean
centerline file exactly as it already does for mark_rows.py output.

polygon_to_centerline.py requires each closed row to have an EVEN point
count (pairs of width-defining corners per cross-section -- see its own
docstring); annotate_plots.py's manual polygon mode allows accidental
duplicate-click points (confirmed: one row in the 20250425 survey has a
harmless exact-duplicate vertex from a double-click), so consecutive
duplicate points are dropped here before writing output -- restores the
intended clean quad without touching the source plot_lattice_*.json.

Usage:
  python3 plot_lattice_to_rows.py \
      --plot-lattice plot_lattice_20250425.json \
      --topdown-meta topdown_20250425.yaml \
      --output rows_20250425_manual.yaml
"""

import argparse

import yaml

from farmsim_farm.topdown_projection import pixel_to_world


def _dedupe_consecutive(points_px, tol=1e-6):
    out = []
    for p in points_px:
        if out and abs(p[0] - out[-1][0]) < tol and abs(p[1] - out[-1][1]) < tol:
            continue
        out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-lattice', required=True, help='plot_lattice_*.json from annotate_plots.py')
    parser.add_argument('--topdown-meta', required=True, help='topdown_*.yaml from capture_topdown.py')
    parser.add_argument('--output', required=True, help='Path to write the rows.yaml (mark_rows.py schema)')
    args = parser.parse_args()

    with open(args.plot_lattice) as f:
        lattice = yaml.safe_load(f) if args.plot_lattice.endswith(('.yaml', '.yml')) else __import__('json').load(f)
    with open(args.topdown_meta) as f:
        meta = yaml.safe_load(f)

    manual_polygons = lattice.get('manual_polygons_px', [])
    if not manual_polygons:
        raise SystemExit(f'{args.plot_lattice} has no manual_polygons_px -- nothing to convert')

    out_rows = []
    n_deduped = 0
    for i, poly in enumerate(manual_polygons):
        points_px = _dedupe_consecutive(poly['points_px'])
        if len(points_px) != len(poly['points_px']):
            n_deduped += 1
        if len(points_px) % 2 != 0:
            raise SystemExit(
                f'Row {i}: {len(points_px)} point(s) after de-duplication (odd) -- '
                'polygon_to_centerline.py needs an even count; inspect this polygon by hand')

        world_points = [list(pixel_to_world(u, v, meta)) for (u, v) in points_px]
        out_rows.append({'id': i, 'points': [[round(x, 3), round(y, 3)] for x, y in world_points], 'closed': True})

    with open(args.output, 'w') as f:
        yaml.safe_dump({'rows': out_rows}, f, default_flow_style=False)
    print(f'Wrote {len(out_rows)} row polygon(s) to {args.output} '
          f'({n_deduped} row(s) had duplicate-click points removed)')


if __name__ == '__main__':
    main()
