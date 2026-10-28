"""
polygon_to_centerline.py

Converts the closed row polygons you draw in mark_rows.py (press 'c') into
open centerline polylines row_follower.py/row_geometry.py can actually
drive: those consumers just walk a row's points in order as a path, so
feeding them a closed rectangle-ish polygon directly makes the "path"
zigzag out to a corner and back rather than following the lane's middle.

Assumes each closed row was drawn as a simple rectangle-like quad, walked
around its perimeter -- 2 points per "end" (a short edge across the
lane's width), alternating with 2 long edges down its length. That's
what mark_rows.py naturally produces if you click the four corners of a
lane in order. Concretely: points are grouped in consecutive pairs
(0,1), (2,3), ... and each pair's midpoint becomes one centerline point,
in order -- so a 4-point quad becomes a 2-point centerline (straight
lane), a 6-point hexagon-ish shape becomes a 3-point centerline (one
bend), etc. Requires an even point count per closed row. Open rows
(closed: false, e.g. ones already marked as centerlines, or produced by
segment_rows.py) pass through unchanged.

Usage:
  python3 polygon_to_centerline.py --input rows_20250624.yaml \
      --output rows_centerline_20250624.yaml
"""

import argparse

import numpy as np
import yaml


def polygon_to_centerline(points):
    if len(points) % 2 != 0:
        raise ValueError(
            f'Closed row has {len(points)} points (odd) -- expected an even '
            'count (pairs of width-defining corners per cross-section)')
    pts = np.array(points, dtype=float)
    pairs = pts.reshape(-1, 2, 2)
    midpoints = pairs.mean(axis=1)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in midpoints]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='rows.yaml from mark_rows.py')
    parser.add_argument('--output', required=True, help='Path to write the centerline rows.yaml')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = yaml.safe_load(f)

    out_rows = []
    for row in data.get('rows', []):
        points = row['points']
        if row.get('closed', False):
            points = polygon_to_centerline(points)
        out_rows.append({'id': row['id'], 'points': points})

    with open(args.output, 'w') as f:
        yaml.safe_dump({'rows': out_rows}, f, default_flow_style=False)
    print(f'Wrote {len(out_rows)} centerline row(s) to {args.output}')


if __name__ == '__main__':
    main()
