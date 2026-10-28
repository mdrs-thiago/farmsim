"""
paths.py

A path through the field is an ordered list of row passes, configured in
cfg/paths/*.yaml:

    field: 20250425
    passes:
      - {row: 3, enter: west, exit: east}
      - {row: 5, enter: east, exit: west}

`enter`/`exit` name the row end by compass direction in the world frame
(east = +x, north = +y) or as `start`/`end` (first/last point of the row in
the rows file). They must be opposite ends of the row.

`FieldPath.reference()` turns the passes into one polyline the robot
follows: a lead-in before each row, the row centre line, a lead-out after
it, and a headland transition (lead-out end -> next lead-in start) between
passes. Every segment knows whether it is in-row (strategy under test,
perception allowed) or headland (geometric pure pursuit on ground truth).
"""

from dataclasses import dataclass

import numpy as np
import yaml

from farmsim_farm.row_geometry import normalize_angle

COMPASS = {'east': (1.0, 0.0), 'west': (-1.0, 0.0), 'north': (0.0, 1.0), 'south': (0.0, -1.0)}


@dataclass
class RowPass:
    row: int
    enter: str
    exit: str


def load_rows_by_id(rows_file):
    """{row id: (N,2) array}; ids come from the file's `id` field when present."""
    with open(rows_file) as f:
        data = yaml.safe_load(f)
    rows = {}
    for i, row in enumerate(data.get('rows', [])):
        pts = np.asarray(row['points'], dtype=float)
        if len(pts) >= 2:
            rows[int(row.get('id', i))] = pts
    return rows


def _end_index(points, end):
    """0 for the row's first point, -1 for its last."""
    if end in ('start', 'end'):
        return 0 if end == 'start' else -1
    if end not in COMPASS:
        raise ValueError(f'row end {end!r} must be one of {sorted(COMPASS) + ["start", "end"]}')
    d = np.array(COMPASS[end])
    return 0 if points[0] @ d > points[-1] @ d else -1


@dataclass
class Segment:
    a: np.ndarray
    b: np.ndarray
    kind: str        # 'row' | 'lead' | 'headland'
    pass_index: int
    row: int

    @property
    def length(self):
        return float(np.linalg.norm(self.b - self.a))


class FieldPath:

    def __init__(self, field, passes, rows_file, lead_in_m=2.0, lead_out_m=2.0):
        self.field = str(field)
        self.passes = [p if isinstance(p, RowPass) else RowPass(**p) for p in passes]
        self.rows_file = str(rows_file)
        self.lead_in_m, self.lead_out_m = lead_in_m, lead_out_m
        self.rows = load_rows_by_id(rows_file)
        self.validate()

    @classmethod
    def load(cls, path_yaml, rows_file=None):
        with open(path_yaml) as f:
            cfg = yaml.safe_load(f)
        if rows_file is None:
            from farmsim_farm.fields import Field
            rows_file = Field(cfg['field']).rows_file
        return cls(cfg['field'], cfg['passes'], rows_file,
                   cfg.get('lead_in_m', 2.0), cfg.get('lead_out_m', 2.0))

    def to_dict(self):
        return {'field': self.field, 'lead_in_m': self.lead_in_m, 'lead_out_m': self.lead_out_m,
                'passes': [vars(p) for p in self.passes]}

    def validate(self):
        if not self.passes:
            raise ValueError('a path needs at least one row pass')
        for i, p in enumerate(self.passes):
            if p.row not in self.rows:
                raise ValueError(f'pass {i}: row {p.row} not in rows file '
                                 f'(ids {min(self.rows)}..{max(self.rows)})')
            pts = self.rows[p.row]
            if _end_index(pts, p.enter) == _end_index(pts, p.exit):
                raise ValueError(f'pass {i}: enter={p.enter} and exit={p.exit} are the same end of row {p.row}')

    def oriented_row(self, p):
        pts = self.rows[p.row]
        return pts if _end_index(pts, p.enter) == 0 else pts[::-1]

    def reference(self):
        """List of Segments, in driving order."""
        segs = []
        prev_exit = None
        for i, p in enumerate(self.passes):
            pts = self.oriented_row(p)
            d_in = (pts[1] - pts[0]) / np.linalg.norm(pts[1] - pts[0])
            d_out = (pts[-1] - pts[-2]) / np.linalg.norm(pts[-1] - pts[-2])
            lead_start = pts[0] - d_in * self.lead_in_m
            if prev_exit is not None:
                segs.append(Segment(prev_exit, lead_start, 'headland', i, p.row))
            segs.append(Segment(lead_start, pts[0], 'lead', i, p.row))
            for a, b in zip(pts, pts[1:]):
                segs.append(Segment(a, b, 'row', i, p.row))
            prev_exit = pts[-1] + d_out * self.lead_out_m
            segs.append(Segment(pts[-1], prev_exit, 'lead', i, p.row))
        return [s for s in segs if s.length > 1e-6]

    def spawn_pose(self):
        """(x, y, yaw) at the start of the first lead-in, facing into the row."""
        seg = self.reference()[0]
        d = seg.b - seg.a
        return float(seg.a[0]), float(seg.a[1]), float(np.arctan2(d[1], d[0]))


class PathTracker:
    """Arc-length progress along a FieldPath's reference polyline.

    Progress only moves forward and only searches a short window ahead, so
    the robot is never snapped onto a neighbouring row 1.5 m away, the
    failure that plain nearest-row lookup has in a field of parallel rows.
    The first update searches the whole path, so the robot may start
    anywhere on it (record_dataset.py spawns mid-row); it takes the EARLIEST
    segment within start_radius_m, because where a path comes back along a
    neighbouring row the later segment can be the closer one.
    """

    def __init__(self, path, search_ahead_m=3.0, start_radius_m=1.0):
        self.segs = path.reference()
        self.cum = np.concatenate([[0.0], np.cumsum([s.length for s in self.segs])])
        self.total = float(self.cum[-1])
        self.s = 0.0
        self.search_ahead_m = search_ahead_m
        self.start_radius_m = start_radius_m
        self.initialized = False

    def _project(self, i, p):
        seg = self.segs[i]
        ab = seg.b - seg.a
        t = float(np.clip(np.dot(p - seg.a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        return t, seg.a + t * ab

    def update(self, x, y, yaw):
        """Advance progress for pose (x, y, yaw); returns a dict with the
        current segment kind, row, lateral offset, heading error and the
        remaining distance to the end of the current row pass."""
        p = np.array([x, y], dtype=float)
        best = None
        i0 = min(max(0, int(np.searchsorted(self.cum, self.s, side='right')) - 1), len(self.segs) - 1)
        if not self.initialized:
            cands = []
            for i in range(len(self.segs)):
                t, q = self._project(i, p)
                cands.append((float(np.linalg.norm(p - q)), i, self.cum[i] + t * self.segs[i].length))
            near = [c for c in cands if c[0] <= self.start_radius_m]
            best = near[0] if near else min(cands)
        else:
            for i in range(i0, len(self.segs)):
                if self.cum[i] > self.s + self.search_ahead_m:
                    break
                t, q = self._project(i, p)
                s = self.cum[i] + t * self.segs[i].length
                if s < self.s - 1e-6:
                    continue
                dist = float(np.linalg.norm(p - q))
                if best is None or dist < best[0] - 1e-9:
                    best = (dist, i, s)
        self.initialized = True
        if best is not None:
            _, i, self.s = best
        else:
            i = i0
        seg = self.segs[i]
        d = (seg.b - seg.a) / seg.length
        v = p - seg.a
        lateral = float(d[0] * v[1] - d[1] * v[0])
        heading = float(normalize_angle(np.arctan2(d[1], d[0]) - yaw))
        pass_end = max(self.cum[j + 1] for j, sg in enumerate(self.segs)
                       if sg.pass_index == seg.pass_index and sg.kind == 'row')
        return {'kind': seg.kind, 'row': seg.row, 'pass_index': seg.pass_index,
                'lateral_offset': lateral, 'heading_error': heading,
                'dist_to_end': max(0.0, pass_end - self.s), 's': self.s,
                'done': self.s >= self.total - 0.05}

    def lookahead_point(self, lookahead_m):
        s = min(self.s + lookahead_m, self.total)
        i = min(int(np.searchsorted(self.cum, s, side='right')) - 1, len(self.segs) - 1)
        seg = self.segs[i]
        t = (s - self.cum[i]) / max(seg.length, 1e-9)
        return seg.a + min(t, 1.0) * (seg.b - seg.a)
