"""
annotate_plots.py

Interactive plot annotation tool over the top-down capture, two ways to get
polygons:

  Parametric grid (modes 1/2/3): click the 4 corners of the trial block,
  fit a homography, then tune n_rows/n_cols/alley width/global offset+scale
  /per-row+per-column nudges with a live overlay. Fast, but assumes a
  regular pitch -- only correct when the real plots really are evenly
  spaced.

  Manual polygons (mode 4): click each plot's own corners/outline directly
  (any number of vertices, closed with 'c'), for fields where spacing or
  plot shape isn't regular enough for the grid to represent -- confirmed
  necessary on the Vollebekk 20250425 survey (real plot spacing turned out
  uneven; the grid's fixed pitch couldn't be tuned to match it, however the
  4 corners/nudges were adjusted). Once any manual polygon exists, it wins:
  the grid is not drawn or exported, so the two approaches are never
  silently mixed. The exclude/bare toggle (mode 3) and the block-corner
  click (mode 1, still needed as the GCP reference for Step 4 verification
  -- unrelated to which polygon source is active) work the same way
  against either source.

Grid math lives in plot_lattice.py (pure, unit-tested); this module is
GUI/IO glue: cv2 window/trackbars/mouse handling, and reading/writing the
two on-disk artifacts. Replaces mark_rows.py's role for LABELING geometry
(mark_rows.py itself is unchanged and still used for driving-corridor
marking).

Usage:
  python3 annotate_plots.py --meta /path/to/topdown_20250425.yaml \
      --output-lattice /path/to/plot_lattice_20250425.json \
      --output-geojson /path/to/plot_polygons_20250425.geojson \
      [--load /path/to/existing_plot_lattice_20250425.json]

Controls:
  1            corner mode: click up to 4 points, in order
               top-left, top-right, bottom-right, bottom-left -- always
               needed (GCP reference), regardless of polygon source below
  r            reset corners (start over)
  2            select/nudge mode (grid only): click the plot nearest the
               cursor to select it, then use arrow keys to nudge its row
               (Up/Down) and column (Left/Right)
  3            exclude/bare toggle mode: click a plot to toggle it
               excluded (drawn red instead of its rainbow/white color)
  4            manual polygon mode: click points to outline the current
               plot (any number of vertices); 'c' closes it and starts the
               next one; 'u' undoes the last point, or the last closed
               polygon if the current one is empty
  Trackbars    n_rows, n_cols, alley % (of plot pitch), global offset X/Y
               (%), global scale (%) -- all live, grid mode only (ignored
               once a manual polygon exists)
  s            save (writes both output files; safe to repeat mid-session)
  q / ESC      save and quit
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml

from farmsim_farm.plot_lattice import (
    apply_global_offset_scale,
    apply_nudges,
    fit_block_homography,
    generate_plot_grid,
    plots_to_geojson,
    plots_to_pixel,
)
from farmsim_farm.topdown_projection import pixel_to_world
from farmsim_farm.topdown_projection import load_topdown_meta

_MODE_CORNERS = 1
_MODE_NUDGE = 2
_MODE_EXCLUDE = 3
_MODE_MANUAL_POLYGON = 4

_NUDGE_STEP = 0.001  # normalized block-space units per arrow-key press

# cv2 arrow-key codes vary by platform/build; these cover the common Linux
# GTK/Qt backends. If arrow keys don't respond, WASD is the fallback below.
_ARROW_UP = {82, 65362, 2490368}
_ARROW_DOWN = {84, 65364, 2621440}
_ARROW_LEFT = {81, 65361, 2424832}
_ARROW_RIGHT = {83, 65363, 2555904}

_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
_EXCLUDED_COLOR = (0, 0, 200)
_SELECTED_COLOR = (0, 255, 255)


def _trackbar_or(window, name, default, maximum, on_change):
    cv2.createTrackbar(name, window, default, maximum, on_change)


class PlotAnnotator:

    def __init__(self, meta_path, output_lattice, output_geojson, load_path=None):
        self.meta = load_topdown_meta(meta_path)
        self.output_lattice = output_lattice
        self.output_geojson = output_geojson

        self.image = cv2.imread(self.meta['image_path'])
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.meta['image_path']}")

        self.corners_px = []  # up to 4 (u, v) tuples
        self.row_nudges = {}
        self.col_nudges = {}
        self.excluded = set()  # {(row, col)} -- (0, idx) for manual polygons, see _current_plots_px
        self.mode = _MODE_CORNERS
        self.selected_plot = None  # (row, col) currently selected for nudging

        self.manual_polygons = []  # [{'points_px': [(u, v), ...]}, ...], finished shapes
        self.current_manual_points = []  # in-progress shape

        self.window = 'annotate_plots -- 1=corners 2=nudge 3=exclude 4=manual s=save q=quit'
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        _trackbar_or(self.window, 'n_rows', 10, 200, lambda v: self._redraw())
        _trackbar_or(self.window, 'n_cols', 8, 200, lambda v: self._redraw())
        _trackbar_or(self.window, 'alley_pct', 15, 90, lambda v: self._redraw())
        _trackbar_or(self.window, 'offset_x_pct+50', 50, 100, lambda v: self._redraw())
        _trackbar_or(self.window, 'offset_y_pct+50', 50, 100, lambda v: self._redraw())
        _trackbar_or(self.window, 'scale_pct', 100, 150, lambda v: self._redraw())

        if load_path:
            self._load(load_path)

        self._redraw()

    # -- parameter readback -------------------------------------------------

    def _get_params(self):
        n_rows = max(1, cv2.getTrackbarPos('n_rows', self.window))
        n_cols = max(1, cv2.getTrackbarPos('n_cols', self.window))
        alley_frac = cv2.getTrackbarPos('alley_pct', self.window) / 100.0
        offset_x = (cv2.getTrackbarPos('offset_x_pct+50', self.window) - 50) / 100.0
        offset_y = (cv2.getTrackbarPos('offset_y_pct+50', self.window) - 50) / 100.0
        scale = cv2.getTrackbarPos('scale_pct', self.window) / 100.0
        return n_rows, n_cols, alley_frac, offset_x, offset_y, scale

    def _set_params(self, n_rows, n_cols, alley_frac, offset_x, offset_y, scale):
        cv2.setTrackbarPos('n_rows', self.window, n_rows)
        cv2.setTrackbarPos('n_cols', self.window, n_cols)
        cv2.setTrackbarPos('alley_pct', self.window, round(alley_frac * 100))
        cv2.setTrackbarPos('offset_x_pct+50', self.window, round(offset_x * 100) + 50)
        cv2.setTrackbarPos('offset_y_pct+50', self.window, round(offset_y * 100) + 50)
        cv2.setTrackbarPos('scale_pct', self.window, round(scale * 100))

    # -- grid construction ---------------------------------------------------

    def _current_plots_px(self):
        """Returns (plots_px, H). plots_px entries always carry 'row', 'col',
        'corners_px', 'excluded'; H is the grid's homography (None in manual
        mode, where no homography is involved -- points are already pixel
        coordinates).

        Manual polygons WIN once any exist: this is the single choke point
        _redraw/_nearest_plot/_save all go through, so once populated,
        manual polygons transparently replace the grid everywhere else in
        this class with no further branching needed there.

        Grid mode returns ([], None) before 4 corners are clicked (was
        (None, None) -- callers already treat both as "nothing to draw/
        export" via truthiness, so this unifies the two sources' empty
        case without special-casing either)."""
        if self.manual_polygons:
            plots = []
            for idx, poly in enumerate(self.manual_polygons):
                plots.append({
                    'row': 0, 'col': idx,
                    'corners_px': np.array(poly['points_px'], dtype=np.float64),
                    'excluded': (0, idx) in self.excluded,
                })
            return plots, None

        if len(self.corners_px) != 4:
            return [], None
        n_rows, n_cols, alley_frac, offset_x, offset_y, scale = self._get_params()
        H, _ = fit_block_homography(self.corners_px)
        plots = generate_plot_grid(n_rows, n_cols, alley_frac)
        plots = apply_global_offset_scale(plots, offset=(offset_x, offset_y), scale=scale)
        plots = apply_nudges(plots, row_nudges=self.row_nudges, col_nudges=self.col_nudges)
        for p in plots:
            p['excluded'] = (p['row'], p['col']) in self.excluded
        return plots_to_pixel(plots, H), H

    def _nearest_plot(self, u, v):
        plots_px, _ = self._current_plots_px()
        if not plots_px:
            return None
        best, best_dist = None, float('inf')
        for p in plots_px:
            center = p['corners_px'].mean(axis=0)
            d = float(np.hypot(center[0] - u, center[1] - v))
            if d < best_dist:
                best, best_dist = p, d
        return (best['row'], best['col']) if best is not None else None

    # -- mouse / keyboard -----------------------------------------------------

    def _on_mouse(self, event, u, v, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode == _MODE_CORNERS:
            if len(self.corners_px) < 4:
                self.corners_px.append((float(u), float(v)))
                print(f'  corner {len(self.corners_px)}/4: ({u}, {v})')
                self._save(quiet=True)
        elif self.mode == _MODE_NUDGE:
            self.selected_plot = self._nearest_plot(u, v)
            if self.selected_plot:
                print(f'  selected plot row={self.selected_plot[0]} col={self.selected_plot[1]} '
                      f'-- arrow keys to nudge')
        elif self.mode == _MODE_EXCLUDE:
            plot = self._nearest_plot(u, v)
            if plot:
                if plot in self.excluded:
                    self.excluded.discard(plot)
                    print(f'  plot {plot}: included')
                else:
                    self.excluded.add(plot)
                    print(f'  plot {plot}: excluded/bare')
                self._save(quiet=True)
        elif self.mode == _MODE_MANUAL_POLYGON:
            self.current_manual_points.append((float(u), float(v)))
            print(f'  point {len(self.current_manual_points)} '
                  f"(plot {len(self.manual_polygons)}): ({u}, {v}) -- 'c' to close")
        self._redraw()

    def _handle_key(self, key):
        if key in (ord('1'),):
            self.mode = _MODE_CORNERS
            print('mode: corners')
        elif key in (ord('2'),):
            self.mode = _MODE_NUDGE
            print('mode: select/nudge')
        elif key in (ord('3'),):
            self.mode = _MODE_EXCLUDE
            print('mode: exclude/bare toggle')
        elif key in (ord('4'),):
            self.mode = _MODE_MANUAL_POLYGON
            print('mode: manual polygon -- click points, c=close, u=undo')
        elif key == ord('r'):
            self.corners_px = []
            print('corners reset')
        elif self.mode == _MODE_MANUAL_POLYGON and key == ord('c'):
            if len(self.current_manual_points) >= 3:
                self.manual_polygons.append({'points_px': list(self.current_manual_points)})
                print(f'  plot {len(self.manual_polygons) - 1} closed with '
                      f'{len(self.current_manual_points)} point(s) -- autosaved')
                self.current_manual_points = []
                self._save(quiet=True)  # closing a plot is exactly the moment work becomes
                                         # irreplaceable if lost -- see _save's docstring
            else:
                print(f'  need at least 3 points to close a polygon (have {len(self.current_manual_points)})')
        elif self.mode == _MODE_MANUAL_POLYGON and key == ord('u'):
            if self.current_manual_points:
                self.current_manual_points.pop()
            elif self.manual_polygons:
                removed = self.manual_polygons.pop()
                print(f'  removed plot {len(self.manual_polygons)} '
                      f'({len(removed["points_px"])} point(s)) -- autosaved')
                self._save(quiet=True)
        elif self.mode == _MODE_NUDGE and self.selected_plot is not None:
            row, col = self.selected_plot
            if key in _ARROW_UP or key == ord('w'):
                self.row_nudges[row] = self.row_nudges.get(row, 0.0) - _NUDGE_STEP
            elif key in _ARROW_DOWN:
                self.row_nudges[row] = self.row_nudges.get(row, 0.0) + _NUDGE_STEP
            elif key in _ARROW_LEFT or key == ord('a'):
                self.col_nudges[col] = self.col_nudges.get(col, 0.0) - _NUDGE_STEP
            elif key in _ARROW_RIGHT or key == ord('d'):
                self.col_nudges[col] = self.col_nudges.get(col, 0.0) + _NUDGE_STEP
        self._redraw()

    # -- drawing ---------------------------------------------------------------

    def _redraw(self):
        canvas = self.image.copy()
        for i, (u, v) in enumerate(self.corners_px):
            cv2.circle(canvas, (int(u), int(v)), 10, (255, 255, 255), -1)
            cv2.putText(canvas, str(i), (int(u) + 12, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

        plots_px, _ = self._current_plots_px()
        if plots_px:
            for p in plots_px:
                pts = p['corners_px'].astype(int)
                if p['excluded']:
                    color = _EXCLUDED_COLOR
                elif self.selected_plot == (p['row'], p['col']):
                    color = _SELECTED_COLOR
                else:
                    color = _COLORS[p['col'] % len(_COLORS)]
                cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)

        if self.current_manual_points:
            pts = np.array(self.current_manual_points, dtype=np.int32)
            for pt in pts:
                cv2.circle(canvas, tuple(pt), 4, (255, 255, 255), -1)
            cv2.polylines(canvas, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)

        mode_name = {_MODE_CORNERS: 'CORNERS', _MODE_NUDGE: 'NUDGE', _MODE_EXCLUDE: 'EXCLUDE',
                     _MODE_MANUAL_POLYGON: 'MANUAL'}[self.mode]
        status = f'mode: {mode_name}'
        if self.manual_polygons or self.current_manual_points:
            status += f'  ({len(self.manual_polygons)} plot(s) closed, ' \
                      f'{len(self.current_manual_points)} point(s) in progress)'
        cv2.putText(canvas, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow(self.window, canvas)

    # -- persistence -------------------------------------------------------------

    def _corner_world_points(self):
        return [list(pixel_to_world(u, v, self.meta)) for (u, v) in self.corners_px]

    def _save(self, quiet=False):
        """Block corners are NOT required to save -- they're only needed for
        corners_world_xy (the Phase D GCP reference), which manual polygons
        (already in pixel coordinates, no homography involved) don't depend
        on at all. Gating the whole save on 4 corners used to mean a long
        manual-polygon session could hit 's'/'q' and silently lose every
        polygon drawn if corners hadn't been clicked yet -- confirmed to
        happen for real, not just hypothetically. Now this only ever
        refuses to save when there is truly nothing to write (no corners
        AND no polygons of either kind)."""
        plots_px, _ = self._current_plots_px()
        have_corners = len(self.corners_px) == 4
        if not plots_px and not self.corners_px:
            if not quiet:
                print('Nothing to save yet -- click corners (mode 1) and/or draw a manual '
                      'polygon (mode 4) first.')
            return
        if not have_corners and not quiet:
            print(f'NOTE: saving with {len(self.corners_px)}/4 block corners set -- '
                  'corners_world_xy will be incomplete until all 4 are clicked (mode 1). '
                  'Polygon geometry itself does not need them and is saved in full.')

        n_rows, n_cols, alley_frac, offset_x, offset_y, scale = self._get_params()
        lattice = {
            'survey_meta': os.path.basename(self.meta.get('image_path', '')),
            'corners_px': [list(p) for p in self.corners_px],
            'corners_world_xy': self._corner_world_points() if have_corners else [],
            'n_rows': n_rows,
            'n_cols': n_cols,
            'alley_width_fraction': alley_frac,
            'global_offset': [offset_x, offset_y],
            'global_scale': scale,
            'row_nudges': {str(k): v for k, v in self.row_nudges.items()},
            'col_nudges': {str(k): v for k, v in self.col_nudges.items()},
            'excluded_plots': sorted([list(p) for p in self.excluded]),
            # Raw manual polygon vertices -- not derivable from the grid
            # params above, so these need their own persistence. Present
            # (possibly empty) even in pure grid-mode sessions, so _load
            # doesn't need to guess whether an old file predates this mode.
            'manual_polygons_px': [{'points_px': [list(p) for p in poly['points_px']]}
                                    for poly in self.manual_polygons],
        }
        with open(self.output_lattice, 'w') as f:
            json.dump(lattice, f, indent=2)

        if plots_px:
            geojson = plots_to_geojson(plots_px)
            with open(self.output_geojson, 'w') as f:
                json.dump(geojson, f, indent=2)

        if not quiet:
            print(f'Wrote lattice params to {self.output_lattice}')
            if plots_px:
                source = 'manual' if self.manual_polygons else 'grid'
                print(f'Wrote {len(plots_px)} plot polygon(s) ({source}) to {self.output_geojson}')

    def _load(self, load_path):
        with open(load_path, 'r') as f:
            lattice = json.load(f)
        self.corners_px = [tuple(p) for p in lattice['corners_px']]
        self._set_params(lattice['n_rows'], lattice['n_cols'], lattice['alley_width_fraction'],
                          lattice['global_offset'][0], lattice['global_offset'][1],
                          lattice['global_scale'])
        self.row_nudges = {int(k): v for k, v in lattice.get('row_nudges', {}).items()}
        self.col_nudges = {int(k): v for k, v in lattice.get('col_nudges', {}).items()}
        self.excluded = {tuple(p) for p in lattice.get('excluded_plots', [])}
        self.manual_polygons = [{'points_px': [tuple(p) for p in poly['points_px']]}
                                 for poly in lattice.get('manual_polygons_px', [])]
        print(f'Loaded lattice params from {load_path}'
              + (f' ({len(self.manual_polygons)} manual plot(s))' if self.manual_polygons else ''))

    # -- main loop -----------------------------------------------------------

    def run(self):
        print(__doc__)
        while True:
            key = cv2.waitKey(20) & 0xFFFFFF
            if key in (ord('q'), 27):
                self._save()
                break
            elif key == ord('s'):
                self._save()
            elif key != 0xFFFFFF:
                self._handle_key(key)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--meta', required=True, help='Path to topdown_*.yaml from capture_topdown')
    parser.add_argument('--output-lattice', required=True, help='Path to write plot_lattice_*.json')
    parser.add_argument('--output-geojson', required=True, help='Path to write plot_polygons_*.geojson')
    parser.add_argument('--load', default=None, help='Existing plot_lattice_*.json to resume editing')
    args = parser.parse_args()

    tool = PlotAnnotator(args.meta, args.output_lattice, args.output_geojson, load_path=args.load)
    tool.run()


if __name__ == '__main__':
    main()
