#!/usr/bin/env python3
"""
select_path.py -- define, check and preview the path the robot follows.

A path is an ordered list of row passes (enter one end of a row, leave by
the other), stored in cfg/paths/*.yaml and used by run_control.py,
run_perception.py --live and optimize.py (rl).

  # check an existing path and draw it over the field's top-down image
  python select_path.py cfg/paths/example.yaml

  # write a new path: row 3 west->east, then row 5 east->west
  python select_path.py --field 20250425 --passes 3:west:east 5:east:west --out cfg/paths/two_rows.yaml

  # list the field's rows and their end points
  python select_path.py --field 20250425 --list-rows

The preview goes to outputs/paths/<name>.png.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from farmsim_farm.fields import Field
from farmsim_farm.paths import FieldPath, RowPass, load_rows_by_id
from farmsim_farm.topdown_projection import world_to_pixel
from farmsim_utils.paths import OUTPUTS_DIR
from farmsim_utils.pipeline import repo_path

COLORS = {'row': (40, 200, 40), 'lead': (0, 200, 255), 'headland': (255, 80, 0)}


def parse_pass(text):
    try:
        row, enter, exit_ = text.split(':')
        return RowPass(int(row), enter, exit_)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{text!r}: expected ROW:ENTER:EXIT, e.g. 3:west:east')


def list_rows(field):
    rows = load_rows_by_id(field.rows_file)
    print(f'{len(rows)} rows in {field.rows_file}')
    for rid, pts in rows.items():
        length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        print(f'  row {rid:3d}: ({pts[0][0]:7.2f}, {pts[0][1]:7.2f}) -> ({pts[-1][0]:7.2f}, {pts[-1][1]:7.2f})'
              f'  {length:5.1f} m')


def draw(path, field, out_png, margin_m=6.0, max_px=1600):
    meta = field.topdown_meta
    img = cv2.imread(str(field.topdown_image))
    if img is None:
        print(f'(no top-down image for field {field.date}, skipping preview)')
        return None
    segs = path.reference()
    to_px = lambda p: np.array(world_to_pixel(p[0], p[1], meta), dtype=float)  # noqa: E731
    pts = np.array([to_px(p) for s in segs for p in (s.a, s.b)])
    for rid, row in path.rows.items():
        cv2.polylines(img, [np.array([to_px(p) for p in row], np.int32)], False, (200, 200, 200), 2)
        a = to_px(row[0])
        cv2.putText(img, str(rid), (int(a[0]) - 30, int(a[1]) + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for s in segs:
        cv2.arrowedLine(img, tuple(to_px(s.a).astype(int)), tuple(to_px(s.b).astype(int)),
                        COLORS[s.kind], 5, tipLength=0.02)
    x, y, _ = path.spawn_pose()
    cv2.circle(img, tuple(to_px((x, y)).astype(int)), 14, (0, 0, 255), -1)
    # crop around the path (plus margin)
    mpp = abs(meta['camera_pose']['z'] - meta['ground_plane_z']) / meta['intrinsics']['fx']
    m = margin_m / mpp
    x0, y0 = np.maximum(pts.min(0) - m, 0).astype(int)
    x1, y1 = np.minimum(pts.max(0) + m, [img.shape[1], img.shape[0]]).astype(int)
    crop = img[y0:y1, x0:x1]
    scale = min(1.0, max_px / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), crop)
    return out_png


def describe(path):
    segs = path.reference()
    total = sum(s.length for s in segs)
    print(f'field {path.field}, {len(path.passes)} pass(es), {total:.1f} m total')
    for i, p in enumerate(path.passes):
        row_len = sum(s.length for s in segs if s.pass_index == i and s.kind == 'row')
        head = sum(s.length for s in segs if s.pass_index == i and s.kind == 'headland')
        extra = f', headland before it {head:.1f} m' if head else ''
        print(f'  pass {i}: row {p.row} {p.enter} -> {p.exit}, {row_len:.1f} m{extra}')
    x, y, yaw = path.spawn_pose()
    print(f'  spawn: x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):.1f} deg')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path_file', nargs='?', help='existing path YAML to check')
    ap.add_argument('--field', help='field date (for --passes / --list-rows)')
    ap.add_argument('--passes', nargs='+', type=parse_pass, help='ROW:ENTER:EXIT ...')
    ap.add_argument('--lead-in-m', type=float, default=2.0)
    ap.add_argument('--lead-out-m', type=float, default=2.0)
    ap.add_argument('--out', help='write the path YAML here')
    ap.add_argument('--list-rows', action='store_true')
    ap.add_argument('--no-preview', action='store_true')
    args = ap.parse_args()

    if args.path_file:
        src = repo_path(args.path_file)
        path = FieldPath.load(src)
        name = Path(src).stem
    else:
        if not args.field:
            ap.error('give a path file, or --field')
        field = Field(args.field)
        if args.list_rows:
            list_rows(field)
            return
        if not args.passes:
            ap.error('--passes is required to build a new path')
        path = FieldPath(args.field, args.passes, field.rows_file, args.lead_in_m, args.lead_out_m)
        name = Path(args.out).stem if args.out else 'path'
    describe(path)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            yaml.safe_dump(path.to_dict(), f, sort_keys=False, default_flow_style=None)
        print(f'wrote {out}')
    if not args.no_preview:
        png = draw(path, Field(path.field), OUTPUTS_DIR / 'paths' / f'{name}.png')
        if png:
            print(f'preview: {png}')


if __name__ == '__main__':
    sys.exit(main())
