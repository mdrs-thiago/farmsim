"""
mark_rows.py

Standalone (non-ROS) interactive tool: click points along each crop row
(open polyline) or around each row/bed's outline (closed polygon) on the
top-down image captured by capture_topdown.py, converts each click to
world (x, y), and writes a rows.yaml. Open polylines are usable as-is by
row_geometry.py / row_follower.py; closed polygons are for cases where
you want the row's actual footprint (e.g. for a segmentation mask),
carrying a `closed: true` flag row_geometry.py doesn't currently look at.

Usage:
  python3 mark_rows.py --meta /path/to/topdown_20250624.yaml \
      --output /path/to/rows_20250624.yaml

Controls:
  left click   add a point to the current shape
  n            finish the current shape as an open polyline, start a new one
  c            close the current shape into a polygon (connects last point
               back to the first) and start a new one
  s            save rows.yaml (safe to do repeatedly, e.g. mid-session)
  u            undo the last point in the current shape
  q / ESC      save and quit

Math: converts a pixel (u, v) to world (x, y) via topdown_projection.py's
pixel_to_world (single shared implementation, also used by segment_rows.py
and annotate_plots.py) -- a ray-cast from the known static camera pose
through the pinhole model, intersected with a flat ground plane at the
mesh's mean elevation (ground_plane_z in the meta file). See that module's
docstring for why this is an approximation (terrain relief across the field
can introduce some error near the edges), acceptable for marking approximate
row/bed boundaries.
"""

import argparse

import cv2
import yaml

from farmsim_farm.topdown_projection import pixel_to_world
from farmsim_farm.topdown_projection import load_topdown_meta


class RowMarker:

    def __init__(self, meta_path, output_path):
        self.meta = load_topdown_meta(meta_path)

        self.output_path = output_path
        self.image = cv2.imread(self.meta['image_path'])
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.meta['image_path']}")

        # Each point is stored as (u, v, x, y) so the image can be redrawn
        # from the pixel coords without needing an inverse projection.
        self.rows = []          # finished shapes: list of (points, closed)
        self.current_row = []   # in-progress shape: list of (u, v, x, y)
        self.window = 'mark_rows -- click points, n=new row, c=close polygon, s=save, q=quit'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = pixel_to_world(u, v, self.meta)
            self.current_row.append((u, v, x, y))
            print(f'  point: pixel=({u},{v}) -> world=({x:.2f}, {y:.2f})')
            self._redraw()

    def _finish_current(self, closed):
        if self.current_row:
            self.rows.append((self.current_row, closed))
            kind = 'polygon' if closed else 'row'
            print(f'{kind.capitalize()} {len(self.rows) - 1} finished with {len(self.current_row)} points')
            self.current_row = []
            self._redraw()

    def _redraw(self):
        canvas = self.image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        for i, (row, closed) in enumerate(self.rows):
            self._draw_row(canvas, row, colors[i % len(colors)], closed)
        self._draw_row(canvas, self.current_row, (255, 255, 255), False)
        cv2.imshow(self.window, canvas)

    def _draw_row(self, canvas, row, color, closed):
        pts_px = [(u, v) for (u, v, x, y) in row]
        for p in pts_px:
            cv2.circle(canvas, p, 6, color, -1)
        for a, b in zip(pts_px, pts_px[1:]):
            cv2.line(canvas, a, b, color, 2)
        if closed and len(pts_px) > 2:
            cv2.line(canvas, pts_px[-1], pts_px[0], color, 2)

    def run(self):
        print('Controls: left-click=add point, n=new row (open), '
              'c=close as polygon, u=undo, s=save, q/ESC=save+quit')
        self._redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('n'):
                self._finish_current(closed=False)
            elif key == ord('c'):
                self._finish_current(closed=True)
            elif key == ord('u'):
                if self.current_row:
                    self.current_row.pop()
                    self._redraw()
            elif key == ord('s'):
                self._save()
            elif key in (ord('q'), 27):
                self._finish_current(closed=False)
                self._save()
                break
        cv2.destroyAllWindows()

    def _save(self):
        data = {
            'rows': [
                {
                    'id': i,
                    'points': [[round(x, 3), round(y, 3)] for (u, v, x, y) in row],
                    'closed': closed,
                }
                for i, (row, closed) in enumerate(self.rows)
            ]
        }
        with open(self.output_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        print(f'Saved {len(self.rows)} shape(s) to {self.output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta', required=True, help='Path to topdown_*.yaml from capture_topdown')
    parser.add_argument('--output', required=True, help='Path to write rows_*.yaml')
    args = parser.parse_args()

    marker = RowMarker(args.meta, args.output)
    marker.run()


if __name__ == '__main__':
    main()
