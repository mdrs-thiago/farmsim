"""
row_geometry.py

Shared math for the marked-row ground truth (thorvald_navigator.py has no
equivalent -- it's a point-to-point waypoint PID with no cross-track/line-
following logic to reuse, see the plan). Loads rows.yaml (from
mark_rows.py) and computes lateral offset + heading error from a robot
pose to the nearest marked row -- used by row_follower.py for control, and
intended to be reused later by whatever computes per-frame training labels.
"""

import numpy as np
import yaml


def load_rows(yaml_path):
    """Returns a list of rows, each a list of numpy (x, y) points, in the
    order they were marked in mark_rows.py. Rows with fewer than 2 points
    (degenerate, can't define a direction) are skipped."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    rows = []
    for row in data.get('rows', []):
        points = [np.array(p, dtype=float) for p in row['points']]
        if len(points) >= 2:
            rows.append(points)
    return rows


def normalize_angle(angle):
    """Wrap angle to (-pi, pi]. Public (row_follower.py's pure-pursuit
    steering law needs it too, not just nearest_row_offset internally)."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle <= -np.pi:
        angle += 2.0 * np.pi
    return angle


def _closest_point_on_segment(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-9:
        return a
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return a + t * ab


def resolve_direction(d_unit, yaw):
    """+1.0 if the segment direction `d_unit` points the same way the robot
    is heading, -1.0 if it points the opposite way.

    rows.yaml stores each row as the operator happened to draw it, and in
    rows_centerline_20250425.yaml adjacent rows alternate: row 0 runs -x,
    row 1 runs +x, and so on. Taking the stored direction literally means a
    robot driving down row 1 that briefly gets closer to row 0 sees its
    heading_error snap by ~180 deg and its lateral_offset sign invert -- the
    controller then steers hard the wrong way, which is exactly what the
    "bouncing" in the recorded videos was (every nearest-row flip in the
    logs coincided with a >90 deg heading_error jump; 22 of 40 sessions).

    A crop row is a line, not an arrow: driving it "backwards" is just
    driving it. So both nearest_row_offset and find_lookahead_point resolve
    each row to whichever of +/-d faces the robot, which bounds
    |heading_error| at 90 deg and makes the sign conventions continuous
    across a row switch."""
    return 1.0 if (d_unit[0] * np.cos(yaw) + d_unit[1] * np.sin(yaw)) >= 0.0 else -1.0


def _closest_point_on_row(position, points):
    """Same point-to-segment math nearest_row_offset uses, scoped to a
    single already-chosen row, exposing which segment it landed on --
    needed by find_lookahead_point to know where to start walking
    forward from. Returns (seg_index, closest_point)."""
    p = np.array(position, dtype=float)
    best = None
    for i, (a, b) in enumerate(zip(points, points[1:])):
        closest = _closest_point_on_segment(p, a, b)
        dist = np.linalg.norm(p - closest)
        if best is None or dist < best[0]:
            best = (dist, i, closest)
    _, seg_index, closest_point = best
    return seg_index, closest_point


def find_lookahead_point(position, yaw, rows, row_id, lookahead_m):
    """Walk along rows[row_id]'s polyline from the point closest to
    `position`, accumulating arc length, until `lookahead_m` is covered --
    the target point row_follower.py's pure-pursuit steering law steers
    toward. Interpolates within whichever segment covers the remaining
    distance. If the row runs out first, extrapolates past its final point
    along the last segment's direction (a lookahead target must always
    exist, even right at a row's end, rather than returning an already-
    passed point).

    `yaw` decides which way "forward" is: the polyline is walked in the
    direction resolved by resolve_direction(), not blindly toward
    points[-1]. Without this, driving a row that happens to be stored
    back-to-front puts the lookahead target *behind* the robot and
    pure_pursuit steers away from the row.
    """
    points = rows[row_id]
    seg_index, current = _closest_point_on_row(position, points)

    seg = points[seg_index + 1] - points[seg_index]
    seg_norm = np.linalg.norm(seg)
    direction = 1.0 if seg_norm < 1e-9 else resolve_direction(seg / seg_norm, yaw)

    # The polyline vertices still ahead of `current`, in travel order, plus
    # the two points defining the direction to extrapolate along if we run
    # off the end.
    if direction > 0.0:
        ahead = points[seg_index + 1:]
        tail_from, tail_to = points[-2], points[-1]
    else:
        ahead = points[seg_index::-1]
        tail_from, tail_to = points[1], points[0]

    remaining = lookahead_m
    for b in ahead:
        seg_vec = b - current
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-9:
            current = b
            continue
        if seg_len >= remaining:
            return current + seg_vec / seg_len * remaining
        remaining -= seg_len
        current = b

    seg_vec = tail_to - tail_from
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-9:
        return tail_to
    return tail_to + seg_vec / seg_len * remaining


def nearest_row_offset(position, yaw, rows):
    """
    position: (x, y) robot position (e.g. from /odom -- ground truth in sim).
    yaw: robot heading, radians.
    rows: as returned by load_rows().

    Which segment is "nearest" is decided by ordinary point-to-segment
    (clamped) distance; the returned lateral_offset/heading_error are then
    computed against that segment's *infinite* line (not clamped) so the
    signal stays sane near a row's endpoints -- this is what "center row"
    means operationally: whichever marked row line is currently closest.

    Returns (lateral_offset, heading_error, row_id):
      lateral_offset: signed perpendicular distance (m), positive = robot
        is left of the row's direction of travel.
      heading_error: signed angle (rad) the robot would need to turn to
        align with that row. Bounded to [-pi/2, pi/2].

    The row's "direction of travel" is resolved per call to whichever of
    +/-d faces the robot's heading (see resolve_direction) -- NOT the order
    the row was marked in. Rows in rows_centerline_20250425.yaml alternate
    direction, so using the marked order made heading_error jump ~180 deg
    and lateral_offset invert whenever the nearest row changed.
      row_id: index into `rows`.

    Raises ValueError if `rows` is empty or every segment is degenerate
    (zero length).
    """
    if not rows:
        raise ValueError('No rows provided')

    p = np.array(position, dtype=float)
    best = None  # (dist, lateral_offset, heading_error, row_id)

    for row_id, points in enumerate(rows):
        for a, b in zip(points, points[1:]):
            d = b - a
            seg_len = np.linalg.norm(d)
            if seg_len < 1e-9:
                continue

            closest = _closest_point_on_segment(p, a, b)
            dist = np.linalg.norm(p - closest)
            if best is not None and dist >= best[0]:
                continue

            d_unit = (d / seg_len) * resolve_direction(d / seg_len, yaw)
            v = p - a
            lateral_offset = d_unit[0] * v[1] - d_unit[1] * v[0]
            seg_heading = np.arctan2(d_unit[1], d_unit[0])
            heading_error = normalize_angle(seg_heading - yaw)
            best = (dist, lateral_offset, heading_error, row_id)

    if best is None:
        raise ValueError('All rows had degenerate (zero-length) segments')

    _, lateral_offset, heading_error, row_id = best
    return lateral_offset, heading_error, row_id
