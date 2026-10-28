"""
export_recording.py

Phase 8: package a completed recording (real frames + record_camera_poses.py's
poses.jsonl + render_labels.py's dual-channel label_frames/) into a
trainable dataset. Two parallel outputs from ONE pass over the same
ordered frame list:

  - a rosbag2 (mcap storage) with RGB + vegetation-mask + structure-mask +
    fused-mask topics, using each frame's real timestamp (poses.jsonl's
    own 'stamp' field -- the same ground-truth-paired image timestamp
    record_camera_poses.py already wrote) -- the replayable source of
    truth.
  - a synced rgb.mp4 + per-channel PNG mask sequence (vegetation,
    structure, AND fused) for direct, no-bag-extraction dataset loading.
    "Synced" here means INDEX alignment (frame i of rgb.mp4 <-> mask PNG
    i in every channel), guaranteed by writing all of them from the same
    loop over the same ordered list -- NOT wall-clock fidelity. mp4
    playback fps is a median-interval CONSTANT-RATE approximation for
    convenience only (robust to the occasional stale-pose frame
    record_camera_poses.py already skips); the authoritative per-frame
    timestamps live in manifest.json and the bag.

The fused channel (palettes.fuse_structure_vegetation: background / crop /
row / bare-bed) is computed HERE, downstream of the two independently
rendered channels, not rendered by render_labels.py itself -- it's a pure
function of the other two channels' class-id arrays, so keeping it a
derived view (rather than a third GL-rendered texture) means the fusion
RULE can change later without re-baking or re-rendering anything. Both
raw channels are still exported alongside it, in case you want to change
the fusion rule after the fact rather than re-render.

No chunking: operates over the full, already-continuous poses.jsonl
order end-to-end -- a recurrent/temporal model needs whole sequences,
and clip-length/stride/overlap is a training-time concern, not this
script's.

Mask PNGs (both bag and on-disk) hold RAW class-id bytes (0/1/2), not the
recolored RGB the label_frames/ QC images use -- a training pipeline
wants integers, not colors to decode. Re-deriving class ids from the
RGB-recolored QC PNGs (via palettes.rgb_to_classid) instead of piping
render_labels.py's raw class-id arrays straight through is a deliberate
tradeoff: QC artifacts stay human-viewable (blend_check.py), dataset
artifacts stay compact.

Placed on the ROS side (not tools/label_pipeline/), not the offline
venvs: needs rosbag2_py/sensor_msgs/cv_bridge, which aren't importable in
either offline venv (same environment split record_camera_poses.py
already established for gz-transport/rclpy).

Imports palettes from THIS package (farmsim_farm.palettes), not
tools/label_pipeline/palettes.py directly -- this script only ever runs
via `ros2 run` against the colcon-installed tree, which has no relative
path back to the workspace's tools/ directory. See
thorvald_gz_bringup/palettes.py's own docstring for why it's a vendored
copy rather than a shared import.

Also computed here, per frame, and written into manifest.json only (not
the bag/mp4 -- these are training labels, not imagery):
  - lateral_offset/heading_error/row_id via row_geometry.nearest_row_offset
    against --rows-file, referenced to BASE_LINK: T_world_optical is
    un-composed back through gz_camera.xacro's fixed mount (see
    base_link_from_optical) using this run's camera_height/camera_pitch/
    camera_x from run_params.json. The camera-referenced values are kept
    too, explicitly named *_camera.
  - row_bbox/row_polygon: bounding box + simplified contour polygon of
    the structure-channel bed-class connected component nearest the
    image's bottom-center (the row currently being followed, as opposed
    to any other bed strip visible off to the side). None when no bed
    pixels are visible in a frame, rather than crashing.
  - cmd_vel/joint_states, passed through from record_camera_poses.py's
    poses.jsonl (see that module's docstring).
  - run_params.json (if present next to poses.jsonl, written by
    generate_run_params.py's --output-json) merged into manifest.json's
    top level -- control strategy + gains, camera height/pitch,
    perturb_std, seed, sun direction -- so the dataset is stratifiable by
    condition without re-parsing shell output.

Usage:
  ros2 run farmsim_perception export_recording \
      --recording-dir recordings/phase7_test \
      --world-name vollebekk_farm_20250425 \
      --survey-date 20250425 \
      --session-name phase7_test \
      --rows-file farm/fields/20250425/rows_centerline.yaml
"""

import argparse
import json
import os

import cv2
import numpy as np

from farmsim_farm.camera_geometry import R_SENSOR_TO_OPTICAL, rpy_matrix
from farmsim_farm.palettes import (
    STRUCT_BED, STRUCT_PALETTE, VEG_PALETTE, fuse_structure_vegetation, rgb_to_classid,
)
from farmsim_farm.row_geometry import find_lookahead_point, load_rows, nearest_row_offset

# Fallbacks for recordings written before camera_x became a randomized axis
# (and before record_random_session.sh forwarded any of these three) -- these
# are exactly record_camera_poses.py's own parameter defaults, i.e. what those
# older runs actually composed T_world_optical with.
DEFAULT_CAMERA_HEIGHT = 0.90
DEFAULT_CAMERA_PITCH = 0.15
DEFAULT_CAMERA_X = 0.80
DEFAULT_CAMERA_ROLL = 0.0
DEFAULT_CAMERA_YAW = 0.0


def _load_classids(rec):
    """cv2.imread gives a BGR-ordered array matching palettes.py's
    BGR-labeled tuples directly -- render_labels.py's PNGs were written
    via PIL with one manual channel reversal already applied (see its
    _load_classid_raster docstring), which is exactly what cv2.imwrite
    would have done automatically; reading back with cv2.imread reverses
    it again, recovering the original class-id-labeled tuple values with
    no extra flip here. Returns (rgb, veg_ids, struct_ids, fused_ids)."""
    rgb = cv2.imread(rec['rgb_path'])
    veg_ids = rgb_to_classid(cv2.imread(rec['veg_path']), VEG_PALETTE)
    struct_ids = rgb_to_classid(cv2.imread(rec['struct_path']), STRUCT_PALETTE)
    fused_ids = fuse_structure_vegetation(struct_ids, veg_ids)
    return rgb, veg_ids, struct_ids, fused_ids


def load_sequence(recording_dir):
    """Read poses.jsonl and resolve each frame's real RGB path + both
    dual-channel label paths, in poses.jsonl's original (already
    time-ordered) order. Raises if any referenced file is missing rather
    than silently skipping it -- a silent gap here would break sequence
    continuity for a recurrent model without leaving any trace."""
    poses_path = os.path.join(recording_dir, 'poses.jsonl')
    frames_dir = os.path.join(recording_dir, 'frames')
    veg_dir = os.path.join(recording_dir, 'label_frames', 'vegetation')
    struct_dir = os.path.join(recording_dir, 'label_frames', 'structure')

    sequence = []
    with open(poses_path) as f:
        for line in f:
            rec = json.loads(line)
            base = os.path.splitext(rec['frame'])[0]
            rgb_path = os.path.join(frames_dir, rec['frame'])
            veg_path = os.path.join(veg_dir, base + '_veg.png')
            struct_path = os.path.join(struct_dir, base + '_struct.png')
            for p in (rgb_path, veg_path, struct_path):
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"{rec['frame']}: missing {p} -- run render_labels.py's dual-channel pass "
                        'over this recording first')
            sequence.append({
                'frame': rec['frame'], 'stamp': rec['stamp'], 'pose_age_s': rec.get('pose_age_s'),
                'rgb_path': rgb_path, 'veg_path': veg_path, 'struct_path': struct_path,
                'T_world_optical': rec['T_world_optical'],
                'cmd_vel': rec.get('cmd_vel'), 'cmd_vel_age_s': rec.get('cmd_vel_age_s'),
                'joint_states': rec.get('joint_states'), 'joint_states_age_s': rec.get('joint_states_age_s'),
            })
    if not sequence:
        raise RuntimeError(f'No frames found in {poses_path}')
    return sequence


def base_link_from_optical(t_world_optical, camera_height, camera_pitch, camera_x,
                           camera_roll=0.0, camera_yaw=0.0):
    """Invert gz_camera.xacro's fixed base_link->camera mount to recover the
    robot's own pose from the recorded camera pose.

    record_camera_poses.py composes, for the SAME three extrinsics:
        R_world_optical = R_wb @ R_bc @ R_SENSOR_TO_OPTICAL.T
        t_world_camera  = t_wb + R_wb @ t_bc
    with R_bc = rpy_matrix(camera_roll, camera_pitch, camera_yaw) and
    t_bc = [camera_x, 0, camera_height]. roll/yaw default to 0, which is what
    every session recorded before they became randomized axes used.
    Both are rigid and invertible, so this is exact, not an approximation.

    Returns ((x, y), yaw) for base_link.
    """
    T = np.asarray(t_world_optical, dtype=float)
    r_wo = T[:3, :3]
    t_wc = T[:3, 3]

    r_bc = rpy_matrix(camera_roll, camera_pitch, camera_yaw)
    r_wb = r_wo @ R_SENSOR_TO_OPTICAL @ r_bc.T
    t_wb = t_wc - r_wb @ np.array([camera_x, 0.0, camera_height])

    # base_link's own +x axis, projected into the world XY plane.
    yaw = float(np.arctan2(r_wb[1, 0], r_wb[0, 0]))
    return (float(t_wb[0]), float(t_wb[1])), yaw


def _compute_row_state(t_world_optical, rows, camera_height, camera_pitch, camera_x,
                       camera_roll=0.0, camera_yaw=0.0):
    """lateral_offset/heading_error/row_id for one frame, via the same
    row_geometry math row_follower.py drives on, referenced to BASE_LINK.

    Earlier versions used T_world_optical's translation and optical +Z
    directly and called the difference "a small (~0.8m) along-row
    approximation". That was wrong: the camera sits camera_x ahead of
    base_link along the ROBOT's heading, so the cross-track component of
    that lever arm is camera_x * sin(heading_error), not zero, and it is
    only "along-row" when the robot is already perfectly aligned. Measured
    over 6,419 frames of the recorded campaign, |camera-referenced minus
    base_link-referenced lateral_offset| had mean 0.177 m, median 0.084 m,
    p90 0.553 m, max 1.833 m -- 45% of frames above 0.10 m against a
    ~0.15 m tracking signal. The optical-Z yaw proxy is likewise only
    exact at zero chassis roll.

    Returns a dict; every value is None if the nearest-row lookup fails
    (e.g. rows list somehow empty -- load_rows already guards against that
    upstream, this is just defensive). The camera-referenced values are
    kept alongside under explicit *_camera names, since that is what the
    image actually sees and a vision model regressing from the image may
    legitimately want them.
    """
    T = np.asarray(t_world_optical, dtype=float)
    pos, yaw = base_link_from_optical(T, camera_height, camera_pitch, camera_x,
                                      camera_roll, camera_yaw)
    base_pose = (pos, yaw)

    cam_pos = (float(T[0, 3]), float(T[1, 3]))
    fwd = T[:3, 2]
    cam_yaw = float(np.arctan2(fwd[1], fwd[0]))

    try:
        lateral_offset, heading_error, row_id = nearest_row_offset(pos, yaw, rows)
        lat_cam, head_cam, row_cam = nearest_row_offset(cam_pos, cam_yaw, rows)
    except ValueError:
        return {k: None for k in (
            'base_pos', 'base_yaw',
            'lateral_offset', 'lateral_offset_rows', 'heading_error', 'row_id',
            'lateral_offset_camera', 'heading_error_camera', 'row_id_camera')}

    return {
        'base_pos': [float(pos[0]), float(pos[1])],
        'base_yaw': float(yaw),
        'lateral_offset': float(lateral_offset),
        # Same quantity in row-spacings. Metres depend on scene scale the model
        # cannot see; this is the unit a controller actually reasons in.
        'lateral_offset_rows': float(lateral_offset / ROW_SPACING_M),
        'heading_error': float(heading_error),
        'row_id': int(row_id),
        'lateral_offset_camera': float(lat_cam),
        'heading_error_camera': float(head_cam),
        'row_id_camera': int(row_cam),
    }


# Mean perpendicular distance between consecutive row centrelines in
# rows_centerline_20250425.yaml (1.565 m, std 0.053 -- the field is regular).
# Used to express lateral_offset in a scale-free unit: a single image cannot
# pin down absolute metres without knowing the camera height, which is itself
# randomized per session, but "fraction of a row spacing" is a property of the
# scene rather than of the camera, so it is both learnable and directly
# meaningful to a controller (0.5 = halfway to the neighbouring row).
ROW_SPACING_M = 1.565


def load_camera_info(recording_dir):
    """fx/fy/cx/cy/width/height written by record_camera_poses. Returns None if
    absent, in which case the row-line label is skipped rather than guessed --
    focal length is randomized per session now, so assuming a default would put
    the line in the wrong place."""
    path = os.path.join(recording_dir, 'camera_info.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _compute_row_line(t_world_optical, rows, row_id, cam, base_pos, base_yaw,
                      near_m=1.0, far_m=5.0):
    """The LINE TO FOLLOW, as the two points a pursuit controller actually uses:
    the point on the followed row `near_m` ahead of the robot, and the one
    `far_m` ahead, both projected into the image.

    An earlier version sampled the row at two fixed IMAGE rows (the bottom edge
    and 60% up). That was wrong in a way the numbers hid: the row's centreline
    only projects between the vanishing point and the nearest visible ground
    point -- on a typical frame, v from about 171 to 360 -- so asking for the
    line at v = 479 extrapolated the fitted 2D line far past where the row
    physically exists, and asking for it near v = 0 landed above the horizon
    altogether. The fit residual was 0.000 px, so the label looked perfect
    while pointing at nothing.

    Anchoring to ground distance instead keeps both points on real row geometry,
    makes them independent of image size and field of view, and gives the pair a
    direct control meaning: the near point is the cross-track term, the near->far
    tilt is the heading term. `in_frame` flags say whether each point is actually
    visible, which matters because a badly off-track robot genuinely cannot see
    the row it should be on.
    """
    T = np.asarray(t_world_optical, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']
    W, H = cam['width'], cam['height']

    try:
        p_near = find_lookahead_point(base_pos, base_yaw, rows, row_id, near_m)
        p_far = find_lookahead_point(base_pos, base_yaw, rows, row_id, far_m)
    except (IndexError, ValueError):
        return None

    out = []
    for P2 in (p_near, p_far):
        Pc = R.T @ (np.array([P2[0], P2[1], 0.0]) - t)
        if Pc[2] <= 0.05:
            return None  # behind the camera: no meaningful image point
        out.append((fx * Pc[0] / Pc[2] + cx, fy * Pc[1] / Pc[2] + cy))
    (un, vn), (uf, vf) = out

    return {
        'row_line_px': [float(un), float(vn), float(uf), float(vf)],
        # Normalised to [-1, 1] across width/height so the target does not
        # depend on resolution or focal length.
        'row_line_norm': [float(2.0 * un / W - 1.0), float(2.0 * vn / H - 1.0),
                          float(2.0 * uf / W - 1.0), float(2.0 * vf / H - 1.0)],
        'row_line_in_frame': [bool(0 <= un < W and 0 <= vn < H),
                              bool(0 <= uf < W and 0 <= vf < H)],
        # Tilt of the near->far segment, degrees: 0 = straight up the image,
        # positive = leaning right. v grows downward, hence (vn - vf).
        'row_line_angle_deg': float(np.degrees(np.arctan2(uf - un, vn - vf))),
        'row_line_lookahead_m': [near_m, far_m],
    }


def _extract_row_region(struct_ids, near_field_frac=0.35):
    """Bounding box + simplified contour polygon of the bed-class
    connected component nearest the image's bottom-center -- "the row
    currently being followed", as opposed to any other bed strip visible
    off to the side of the frame. Returns ([x0,y0,x1,y1], [[x,y],...]),
    or (None, None) when no bed pixels are visible (extreme turns,
    coverage gaps) rather than raising.

    Connected-component search is restricted to the bottom
    `near_field_frac` of the image before labeling (bbox/polygon are
    still reported in full-image coordinates) -- confirmed empirically
    (scanline gap-width check on a real rendered frame): by ~35% up from
    the bottom, inter-row furrow gaps are still 1px wide (below the
    mesh/raster's resolution at that distance) and adjacent rows' bed
    masks merge into one blob spanning nearly the whole frame width;
    closer than that, gaps are consistently 10-100+ px, comfortably
    separable. The far field isn't usefully separable into individual
    rows this way regardless, so excluding it costs nothing a tight box
    could have captured anyway. Erosion below is a secondary safety net
    for smaller residual bridges within the kept region, not the primary
    fix -- it alone wasn't enough to separate genuinely sub-pixel gaps."""
    mask = (struct_ids == STRUCT_BED).astype(np.uint8)
    if not mask.any():
        return None, None

    h, w = mask.shape
    near_field_top = int(round(h * (1.0 - near_field_frac)))
    search_mask = mask.copy()
    search_mask[:near_field_top, :] = 0
    if not search_mask.any():
        return None, None

    # Empirically, adjacent rows' bed masks are still spuriously bridged
    # even within the near field -- a handful of pixels wide, from
    # antialiasing/relief-detection noise along the furrow line, not a
    # real merge (confirmed: visually-separated strips in the real render
    # came back as one connected component spanning the full frame
    # width). Erode first to sever thin bridges before labeling, then
    # dilate the chosen component back by the same amount (intersected
    # with the true mask) to recover its real extent -- a real furrow
    # gap is much wider than this kernel, so dilation won't re-bridge
    # into a neighboring row, only restore the chosen one's own shape.
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(search_mask, kernel, iterations=1)
    if not eroded.any():
        return None, None

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n_labels <= 1:  # label 0 is background only
        return None, None

    bottom_center = np.array([w / 2.0, float(h - 1)])
    best_label, best_dist = None, None
    for label in range(1, n_labels):
        dist = float(np.hypot(*(centroids[label] - bottom_center)))
        if best_dist is None or dist < best_dist:
            best_label, best_dist = label, dist

    seed = (labels == best_label).astype(np.uint8)
    component_mask = cv2.dilate(seed, kernel, iterations=1) & search_mask

    ys, xs = np.nonzero(component_mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bbox, None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
    return bbox, polygon


def compute_derived_labels(sequence, rows, camera_height=DEFAULT_CAMERA_HEIGHT,
                           camera_pitch=DEFAULT_CAMERA_PITCH, camera_x=DEFAULT_CAMERA_X,
                           camera_roll=DEFAULT_CAMERA_ROLL, camera_yaw=DEFAULT_CAMERA_YAW,
                           camera_info=None):
    """Adds lateral_offset/heading_error/row_id/row_bbox/row_polygon to
    each sequence entry in place -- pose-derived fields need no image
    load at all; row_bbox/row_polygon re-reads just the (small) structure
    mask PNG, a third read of that file alongside write_bag's and
    write_mp4_and_masks's own reads -- simple and correct beats a shared
    decode cache here, these are 640x480 PNGs."""
    for rec in sequence:
        rec.update(_compute_row_state(
            rec['T_world_optical'], rows, camera_height, camera_pitch, camera_x,
            camera_roll, camera_yaw))

        line = None
        if camera_info is not None and rec.get('row_id') is not None:
            line = _compute_row_line(rec['T_world_optical'], rows, rec['row_id'],
                                     camera_info, rec['base_pos'], rec['base_yaw'])
        for k in ('row_line_px', 'row_line_norm', 'row_line_in_frame',
                  'row_line_angle_deg', 'row_line_lookahead_m'):
            rec[k] = line[k] if line else None

        struct_ids = rgb_to_classid(cv2.imread(rec['struct_path']), STRUCT_PALETTE)
        row_bbox, row_polygon = _extract_row_region(struct_ids)
        rec['row_bbox'] = row_bbox
        rec['row_polygon'] = row_polygon


def load_run_params(recording_dir):
    """generate_run_params.py's --output-json companion file, if present
    next to poses.jsonl -- merged into manifest.json's top level so the
    dataset is stratifiable by condition (control strategy + gains,
    camera height/pitch, perturb_std, seed, sun direction) without
    re-parsing run_params.txt's shell-eval-oriented KEY="VALUE" lines.
    Returns {} (not an error) if the run predates --output-json."""
    path = os.path.join(recording_dir, 'run_params.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _stamp_to_ros_time(stamp):
    sec = int(stamp)
    nanosec = int(round((stamp - sec) * 1e9))
    return sec, nanosec


def write_bag(bag_dir, sequence):
    # ROS dependencies live here rather than at module import time so the pure
    # projection/label functions above are reusable by offline repair tools.
    import rclpy.serialization
    from cv_bridge import CvBridge
    from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata

    bridge = CvBridge()
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=bag_dir, storage_id='mcap'), ConverterOptions('', ''))
    for topic_id, (name, msg_type) in enumerate([
            ('/frame/rgb', 'sensor_msgs/msg/Image'),
            ('/frame/vegetation_mask', 'sensor_msgs/msg/Image'),
            ('/frame/structure_mask', 'sensor_msgs/msg/Image'),
            ('/frame/fused_mask', 'sensor_msgs/msg/Image')]):
        writer.create_topic(TopicMetadata(id=topic_id, name=name, type=msg_type, serialization_format='cdr'))

    for rec in sequence:
        sec, nanosec = _stamp_to_ros_time(rec['stamp'])
        timestamp_ns = int(round(rec['stamp'] * 1e9))

        rgb, veg_ids, struct_ids, fused_ids = _load_classids(rec)

        for topic, img, encoding in (
                ('/frame/rgb', rgb, 'bgr8'),
                ('/frame/vegetation_mask', veg_ids, 'mono8'),
                ('/frame/structure_mask', struct_ids, 'mono8'),
                ('/frame/fused_mask', fused_ids, 'mono8')):
            msg = bridge.cv2_to_imgmsg(img, encoding=encoding)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = rec['frame']
            writer.write(topic, rclpy.serialization.serialize_message(msg), timestamp_ns)

    writer.close()


def write_mp4_and_masks(session_dir, sequence, fps):
    mask_dirs = {name: os.path.join(session_dir, 'masks', name)
                 for name in ('vegetation', 'structure', 'fused')}
    for d in mask_dirs.values():
        os.makedirs(d, exist_ok=True)

    first = cv2.imread(sequence[0]['rgb_path'])
    height, width = first.shape[:2]
    video_writer = cv2.VideoWriter(os.path.join(session_dir, 'rgb.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, rec in enumerate(sequence):
        rgb, veg_ids, struct_ids, fused_ids = _load_classids(rec)
        video_writer.write(rgb)
        cv2.imwrite(os.path.join(mask_dirs['vegetation'], f'frame_{i:06d}.png'), veg_ids)
        cv2.imwrite(os.path.join(mask_dirs['structure'], f'frame_{i:06d}.png'), struct_ids)
        cv2.imwrite(os.path.join(mask_dirs['fused'], f'frame_{i:06d}.png'), fused_ids)

    video_writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recording-dir', required=True)
    parser.add_argument('--world-name', required=True, help='e.g. vollebekk_farm_20250425')
    parser.add_argument('--survey-date', required=True, help='e.g. 20250425')
    parser.add_argument('--session-name', required=True, help='e.g. phase7_test')
    parser.add_argument('--rows-file', required=True,
                         help='rows.yaml this session drove against, for lateral_offset/heading_error labels')
    parser.add_argument('--out-root', default='datasets/sim')
    args = parser.parse_args()

    sequence = load_sequence(args.recording_dir)
    rows = load_rows(args.rows_file)
    run_params = load_run_params(args.recording_dir)
    # Must be the SAME extrinsics record_camera_poses.py composed poses.jsonl
    # with, or base_link_from_optical un-composes with the wrong lever arm.
    compute_derived_labels(
        sequence, rows,
        camera_height=float(run_params.get('camera_height', DEFAULT_CAMERA_HEIGHT)),
        camera_pitch=float(run_params.get('camera_pitch', DEFAULT_CAMERA_PITCH)),
        camera_x=float(run_params.get('camera_x', DEFAULT_CAMERA_X)),
        camera_roll=float(run_params.get('camera_roll', DEFAULT_CAMERA_ROLL)),
        camera_yaw=float(run_params.get('camera_yaw', DEFAULT_CAMERA_YAW)),
        camera_info=load_camera_info(args.recording_dir))

    stamps = np.array([rec['stamp'] for rec in sequence])
    diffs = np.diff(stamps)
    diffs = diffs[diffs > 0]
    nominal_fps = float(1.0 / np.median(diffs)) if len(diffs) else 10.0

    session_dir = os.path.join(args.out_root, f'{args.world_name}__{args.survey_date}', args.session_name)
    os.makedirs(session_dir, exist_ok=True)

    print(f'Writing rosbag2 (mcap) for {len(sequence)} frame(s)...')
    write_bag(os.path.join(session_dir, 'bag'), sequence)

    print(f'Writing rgb.mp4 + mask PNG sequences (nominal {nominal_fps:.2f} fps)...')
    write_mp4_and_masks(session_dir, sequence, nominal_fps)

    manifest = {
        'session_name': args.session_name,
        # Same value as session_name, under the name an LSTM-style sequence
        # dataset loader actually looks for -- one recording is one
        # contiguous temporal sequence (frame order is already guaranteed
        # by load_sequence's poses.jsonl-order read), so this is a label
        # for that fact, not a new grouping.
        'sequence_id': args.session_name,
        'world_name': args.world_name,
        'survey_date': args.survey_date,
        'frame_count': len(sequence),
        'nominal_fps': nominal_fps,
        'run_params': run_params,
        'per_frame': [{'frame_index': i, 'frame': r['frame'], 'stamp': r['stamp'],
                       'pose_age_s': r['pose_age_s'],
                       'lateral_offset': r['lateral_offset'], 'heading_error': r['heading_error'],
                       'row_id': r['row_id'], 'row_bbox': r['row_bbox'], 'row_polygon': r['row_polygon'],
                       'lateral_offset_rows': r['lateral_offset_rows'],
                       'row_line_px': r['row_line_px'],
                       'row_line_norm': r['row_line_norm'],
                       'row_line_in_frame': r['row_line_in_frame'],
                       'row_line_angle_deg': r['row_line_angle_deg'],
                       'row_line_lookahead_m': r['row_line_lookahead_m'],
                       'lateral_offset_camera': r['lateral_offset_camera'],
                       'heading_error_camera': r['heading_error_camera'],
                       'row_id_camera': r['row_id_camera'],
                       'cmd_vel': r['cmd_vel'], 'cmd_vel_age_s': r['cmd_vel_age_s'],
                       'joint_states': r['joint_states'], 'joint_states_age_s': r['joint_states_age_s']}
                      for i, r in enumerate(sequence)],
    }
    with open(os.path.join(session_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Done. Wrote {session_dir}')


if __name__ == '__main__':
    main()
