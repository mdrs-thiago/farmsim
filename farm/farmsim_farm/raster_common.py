"""
raster_common.py

Shared world-XY grid / mesh-height-lookup / pinhole-projection machinery
used by every script that bakes something onto the georeferenced world-XY
raster grid: build_label_raster.py (vegetation), build_structure_raster.py
(height/relief), and check_alignment.py (the real birdseye photo itself,
as a known-content projection target). This used to live inline in
build_label_raster.py's main() -- factored out so those three scripts
share one implementation of "look up this world XY cell's real terrain
height" and "project a world point into a camera" instead of three
slowly-diverging copies.

Two projection entry points, for the two camera situations this pipeline
has: a single STATIC birdseye camera (world_to_pixel, camera_pose given as
roll/pitch/yaw + xyz, as written by capture_topdown.py) and a MOVING
per-frame camera (world_to_pixel_dynamic, camera pose given as a
T_world_optical 4x4, as written by record_camera_poses.py per frame). Both
share the same pinhole-division/bounds-check tail (_project_optical) --
only how the world-frame ray direction gets rotated into the optical
frame differs.
"""

import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def rotation_matrix(roll, pitch, yaw):
    """SDF pose convention (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), same as
    bake_world_mesh.sdf_pose_to_matrix and
    farmsim_farm.topdown_projection.rotation_matrix (this venv can't
    import that ROS-package module directly, hence the separate copy)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _project_optical(x_opt, y_opt, z_opt, fx, fy, cx, cy, width, height):
    """Shared pinhole-division + frame-bounds tail for both projection
    entry points below: given a ray already expressed in the OPTICAL
    frame (Z forward, X right, Y down), divide by depth and check the
    result lands within the image and in front of the camera."""
    valid = z_opt > 0.05
    z_safe = np.where(valid, z_opt, 1.0)
    u = fx * x_opt / z_safe + cx
    v = fy * y_opt / z_safe + cy
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def world_to_pixel(points_w, meta):
    """Project an (N,3) array of world points into a STATIC camera
    described by `meta['intrinsics']` (fx,fy,cx,cy) and
    `meta['camera_pose']` (roll,pitch,yaw,x,y,z) -- the topdown_*.yaml
    sidecar capture_topdown.py writes. Moved verbatim from
    build_label_raster.py.

    Returns (u, v, valid) -- valid is False where the point is behind the
    camera or projects outside the photo's pixel bounds.
    """
    intr = meta['intrinsics']
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    pose = meta['camera_pose']
    r_wc = rotation_matrix(pose['roll'], pose['pitch'], pose['yaw'])
    cam_pos = np.array([pose['x'], pose['y'], pose['z']])

    d_w = points_w - cam_pos
    d_sensor = d_w @ r_wc  # r_wc.T applied via right-multiply for row vectors
    # Sensor (X fwd/Y left/Z up) -> optical (Z fwd/X right/Y down).
    x_opt, y_opt, z_opt = -d_sensor[:, 1], -d_sensor[:, 2], d_sensor[:, 0]

    return _project_optical(x_opt, y_opt, z_opt, fx, fy, cx, cy, meta['width'], meta['height'])


def world_to_pixel_dynamic(points_w, T_world_optical, fx, fy, cx, cy, width, height):
    """Project an (N,3) array of world points into a MOVING camera
    described by a T_world_optical 4x4 (row-major, maps a point already in
    the OPTICAL frame to world coordinates -- record_camera_poses.py's
    per-frame pose, already composed through camera_link and the
    sensor->optical rotation, so no further frame permutation is needed
    here unlike world_to_pixel's sensor-frame input).

    Same row-vector-right-multiply trick as world_to_pixel: for an
    orthogonal rotation R_world_from_optical, right-multiplying a
    world-frame row vector by R (not R.T) gives R.T @ v, i.e. the
    optical-frame vector -- avoids an explicit transpose+matmul per point.

    Returns (u, v, valid), same convention as world_to_pixel.
    """
    T = np.asarray(T_world_optical)
    r_wo = T[:3, :3]
    t_wo = T[:3, 3]

    d_w = points_w - t_wo
    d_opt = d_w @ r_wo
    x_opt, y_opt, z_opt = d_opt[:, 0], d_opt[:, 1], d_opt[:, 2]

    return _project_optical(x_opt, y_opt, z_opt, fx, fy, cx, cy, width, height)


def load_height_lookup(mesh_path):
    """Load a baked world-coordinate mesh (bake_world_mesh.py's output)
    and return a (height_tree, mesh_z) pair for nearest-vertex height
    lookup: height_tree is a cKDTree over vertex XY, mesh_z is the
    parallel array of vertex Z. Same pattern build_label_raster.py used
    inline (~18cm average vertex spacing on the Vollebekk survey mesh --
    dense enough for nearest-vertex to noticeably improve on a single
    global constant without needing full ray-mesh intersection)."""
    m = trimesh.load(mesh_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = list(m.geometry.values())[0]
    height_tree = cKDTree(m.vertices[:, :2])
    mesh_z = m.vertices[:, 2]
    return height_tree, mesh_z


def build_output_grid(georef, resolution):
    """World XY of every cell of a fixed-resolution output grid spanning
    `georef`'s (x0, y0, width_m, height_m) footprint. Row 0 = max world y
    (image-style top-left origin), matching every raster this pipeline
    writes with cv2.imwrite/Image.save -- kept here, once, explicit,
    rather than re-derived per script.

    Returns (xs, ys, grid_x, grid_y, flat_x, flat_y, w_px, h_px).
    """
    x0g, y0g = georef['x0'], georef['y0']
    w_px = int(round(georef['width_m'] / resolution))
    h_px = int(round(georef['height_m'] / resolution))

    xs = x0g + (np.arange(w_px) + 0.5) * resolution
    ys = y0g + georef['height_m'] - (np.arange(h_px) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    return xs, ys, grid_x, grid_y, flat_x, flat_y, w_px, h_px


def lookup_heights(height_tree, mesh_z, flat_x, flat_y, return_dist=False):
    """Nearest-vertex terrain height at each (flat_x, flat_y) world point.
    `return_dist=True` also returns the nearest-vertex distance, which
    build_label_raster.py's photo-projection use case never needed (it's
    already gated by the photo's own valid/void mask) but
    build_structure_raster.py does, to null out cells extrapolated far
    outside the real mesh footprint before they pollute a local low-pass."""
    dist, nearest_idx = height_tree.query(np.column_stack([flat_x, flat_y]), k=1)
    flat_z = mesh_z[nearest_idx]
    if return_dist:
        return flat_z, dist
    return flat_z


def write_raster_georef(georef, out_path, resolution, w_px, h_px):
    """Write `<out_path minus extension>_georef.json`: `georef` plus this
    raster's own pixel-grid parameters (may differ trivially from the
    input georef due to rounding width/height to a whole pixel count), so
    downstream consumers never have to re-derive pixel<->world by hand.
    Returns the path written."""
    raster_georef = dict(georef)
    raster_georef.update({
        'raster_path': out_path,
        'resolution_m_per_px': resolution,
        'raster_width_px': w_px,
        'raster_height_px': h_px,
    })
    out_georef_path = out_path.rsplit('.', 1)[0] + '_georef.json'
    with open(out_georef_path, 'w') as f:
        json.dump(raster_georef, f, indent=2)
    return out_georef_path
