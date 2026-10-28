"""
topdown_projection.py

Single explicit pixel<->world transform for the static top-down capture
(topdown_*.yaml from capture_topdown.py), used by every tool that needs to
convert between image pixels and world (x, y): mark_rows.py, segment_rows.py,
and annotate_plots.py. Kept in exactly one place, per the "one explicit
transform, not scattered" requirement -- extracted from mark_rows.py, where
this math originated (rotation_matrix/pixel_to_world), plus a new
world_to_pixel (inverse pinhole projection) for live preview overlays.

Both directions share the same flat-ground-plane assumption
(meta['ground_plane_z']): a known approximation (terrain relief across the
field can introduce some error near the edges), acceptable for authoring
tools whose output is later re-verified by the mesh-aware rendering pipeline
(render_labels.py / check_alignment.py), which does NOT make this assumption.
"""

import numpy as np


def rotation_matrix(roll, pitch, yaw):
    """SDF pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), mapping a
    vector in the camera's local (REP-103, X-forward/Y-left/Z-up) frame into
    the world frame."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def pixel_to_world(u, v, meta):
    """Cast a ray from the static camera through pixel (u, v) and intersect
    it with the flat ground plane z = meta['ground_plane_z'].

    Sanity check: the image center (u=cx, v=cy) must map to (camera_x,
    camera_y) directly below/along the camera's boresight -- verified by
    construction below (the center ray's world-frame x,y components are
    zero before scaling by t).
    """
    intr = meta['intrinsics']
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    pose = meta['camera_pose']

    # Ray direction in the camera's own (optical) frame: standard pinhole
    # model, +Z into the scene.
    d_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # Optical frame (Z fwd, X right, Y down) -> sensor/link frame
    # (REP-103: X fwd, Y left, Z up): X_sensor=Z_opt, Y_sensor=-X_opt,
    # Z_sensor=-Y_opt.
    d_sensor = np.array([d_opt[2], -d_opt[0], -d_opt[1]])

    r = rotation_matrix(pose['roll'], pose['pitch'], pose['yaw'])
    d_world = r @ d_sensor

    cam_pos = np.array([pose['x'], pose['y'], pose['z']])
    ground_z = meta['ground_plane_z']
    if abs(d_world[2]) < 1e-9:
        raise ValueError('Camera ray is parallel to the ground plane -- check camera_pose')
    t = (ground_z - cam_pos[2]) / d_world[2]
    if t <= 0:
        raise ValueError('Ray points away from the ground plane -- check camera_pose')

    world = cam_pos + t * d_world
    return float(world[0]), float(world[1])


def world_to_pixel(x, y, meta):
    """Inverse of pixel_to_world: project a world point on the flat ground
    plane (z = meta['ground_plane_z']) back into the static camera's pixel
    coordinates. Used for live preview overlays (draw a world-space lattice
    back onto the topdown photo) -- the exact algebraic inverse of the ray
    intersection above, not a re-derivation from a different model.

    Returns (u, v). Does not check whether the point falls in front of the
    camera or within the image bounds -- callers that need that should check
    the returned pixel against the image dimensions themselves.
    """
    intr = meta['intrinsics']
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    pose = meta['camera_pose']

    cam_pos = np.array([pose['x'], pose['y'], pose['z']])
    ground_z = meta['ground_plane_z']
    world = np.array([x, y, ground_z])

    r = rotation_matrix(pose['roll'], pose['pitch'], pose['yaw'])
    # world = cam_pos + t * (r @ d_sensor)  =>  r @ d_sensor = (world - cam_pos) / t
    # Solve for d_sensor (up to the unknown positive scale t) via r^T (r is a
    # pure rotation, so its inverse is its transpose).
    d_world_unscaled = world - cam_pos
    d_sensor_unscaled = r.T @ d_world_unscaled

    # Inverse of d_sensor = (d_opt[2], -d_opt[0], -d_opt[1]):
    # d_opt = (-d_sensor[1], -d_sensor[2], d_sensor[0]), any common positive
    # scale cancels once we divide by the Z (depth) component below.
    d_opt = np.array([-d_sensor_unscaled[1], -d_sensor_unscaled[2], d_sensor_unscaled[0]])
    if d_opt[2] <= 0:
        raise ValueError('World point is behind the camera (or on the ground plane at the '
                          'camera itself) -- check inputs')

    u = d_opt[0] / d_opt[2] * fx + cx
    v = d_opt[1] / d_opt[2] * fy + cy
    return float(u), float(v)


def load_topdown_meta(path):
    """Read a topdown_*.yaml; a relative image_path is taken relative to the
    yaml file (fields keep topdown.png next to topdown.yaml)."""
    import os

    import yaml
    with open(path) as f:
        meta = yaml.safe_load(f)
    image = meta.get('image_path')
    if image and not os.path.isabs(image):
        meta['image_path'] = os.path.join(os.path.dirname(os.path.abspath(path)), image)
    return meta
