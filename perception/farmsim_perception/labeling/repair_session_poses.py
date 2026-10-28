#!/usr/bin/env python3
"""
repair_session_poses.py

Rewrites poses.jsonl for sessions recorded before record_random_session.sh
forwarded the camera extrinsics to record_camera_poses.

The bug: record_random_session.sh randomized camera_height/camera_pitch and
passed them to the launch file (so the SIMULATED camera, and therefore every
captured image, really was at those values) but never passed them on to
record_camera_poses, which silently kept its parameter defaults
camera_height=0.90, camera_pitch=0.15, camera_x_offset=0.80. Every
T_world_optical in poses.jsonl was therefore composed from the true base_link
pose with the WRONG mount transform, so render_labels.py projected every label
mask from a camera that wasn't where the real one was. Measured over 302
sessions: correlation(camera_pitch error, check_alignment Part B dy) = +0.920,
fitted slope 6.1 px/deg, dy spanning -63..+57 px.

The fix is exact, not a re-simulation: the composition
    R_world_optical = R_wb @ pitch(camera_pitch) @ R_SENSOR_TO_OPTICAL.T
    t_world_camera  = t_wb + R_wb @ [camera_x, 0, camera_height]
is rigid and invertible, so un-composing with the wrong extrinsics recovers
the true base_link pose bit-for-bit, and re-composing with the right ones
gives the T_world_optical that should have been written. Verified round-trip
to ~1e-14 m / 1e-14 deg including chassis roll and pitch.

camera_x was a hardcoded 0.80 in gz_camera.xacro for these runs (it only
became a randomized axis afterwards), so it is both the wrong and the right
value here -- it is written into run_params.json so export_recording.py's
base_link recovery uses the correct lever arm rather than falling back to a
default that might drift later.

After running this, the session's label_frames/ must be re-rendered
(render_labels.py) and re-exported (export_recording.py) -- this script only
fixes the pose file they are derived from. tools/repair_sessions.sh does all
three.

Usage:
  python3 tools/repair_session_poses.py recordings/run_2026...  [more dirs...]
  python3 tools/repair_session_poses.py --all
  python3 tools/repair_session_poses.py --all --dry-run
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

# Duplicated from record_camera_poses.py rather than imported: this script runs
# as plain python3 (and from the GL venv in the repair driver), outside the
# sourced ROS environment that module's `import rclpy` needs.
R_SENSOR_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])

# record_camera_poses.py's parameter defaults == what the broken runs used.
WRONG_HEIGHT, WRONG_PITCH, WRONG_X = 0.90, 0.15, 0.80
# gz_camera.xacro's hardcoded camera_x at the time these sessions were recorded.
RECORDED_CAMERA_X = 0.80


def pitch_matrix(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def decompose(T, height, pitch, x):
    """T_world_optical -> (R_world_base, t_world_base), given the mount."""
    T = np.asarray(T, dtype=float)
    r_wb = T[:3, :3] @ R_SENSOR_TO_OPTICAL @ pitch_matrix(pitch).T
    t_wb = T[:3, 3] - r_wb @ np.array([x, 0.0, height])
    return r_wb, t_wb


def compose(r_wb, t_wb, height, pitch, x):
    """(R_world_base, t_world_base) + mount -> T_world_optical."""
    T = np.eye(4)
    T[:3, :3] = r_wb @ pitch_matrix(pitch) @ R_SENSOR_TO_OPTICAL.T
    T[:3, 3] = t_wb + r_wb @ np.array([x, 0.0, height])
    return T


def repair_session(rec_dir, dry_run=False):
    """Returns (status, detail). status is 'repaired', 'skipped' or 'failed'."""
    poses_path = os.path.join(rec_dir, 'poses.jsonl')
    params_path = os.path.join(rec_dir, 'run_params.json')
    backup_path = poses_path + '.preextrinsicsfix'

    if not os.path.exists(poses_path):
        return 'skipped', 'no poses.jsonl'
    if not os.path.exists(params_path):
        return 'skipped', 'no run_params.json (cannot know the true extrinsics)'
    if os.path.exists(backup_path):
        return 'skipped', 'already repaired (backup exists)'

    params = json.load(open(params_path))
    true_h = float(params['camera_height'])
    true_p = float(params['camera_pitch'])
    true_x = float(params.get('camera_x', RECORDED_CAMERA_X))

    records = []
    with open(poses_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return 'skipped', 'poses.jsonl empty'

    max_shift = 0.0
    for rec in records:
        r_wb, t_wb = decompose(rec['T_world_optical'], WRONG_HEIGHT, WRONG_PITCH, WRONG_X)
        fixed = compose(r_wb, t_wb, true_h, true_p, true_x)
        max_shift = max(max_shift, float(np.linalg.norm(
            fixed[:3, 3] - np.asarray(rec['T_world_optical'], dtype=float)[:3, 3])))
        rec['T_world_optical'] = fixed.tolist()
        # Keep the recovered robot pose too: it is what the labels are really
        # referenced to, and having it on disk means downstream tools never
        # have to redo this inversion.
        rec['T_world_base_link'] = compose(r_wb, t_wb, 0.0, 0.0, 0.0).tolist()

    detail = (f'{len(records)} frames, h {WRONG_HEIGHT:.2f}->{true_h:.3f} '
              f'p {WRONG_PITCH:.2f}->{true_p:.3f} x {WRONG_X:.2f}->{true_x:.2f}, '
              f'max camera shift {max_shift:.3f} m')
    if dry_run:
        return 'repaired', detail + ' [dry-run]'

    shutil.copy2(poses_path, backup_path)
    tmp = poses_path + '.tmp'
    with open(tmp, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    os.replace(tmp, poses_path)

    if 'camera_x' not in params:
        params['camera_x'] = true_x
        params['_extrinsics_repaired'] = True
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

    return 'repaired', detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*', help='recording directories to repair')
    ap.add_argument('--all', action='store_true', help='repair every recordings/run_* session')
    ap.add_argument('--recordings-root', default='outputs/recordings')
    ap.add_argument('--dry-run', action='store_true', help='report what would change, write nothing')
    args = ap.parse_args()

    dirs = list(args.dirs)
    if args.all:
        dirs += sorted(glob.glob(os.path.join(args.recordings_root, 'run_*')))
    if not dirs:
        ap.error('pass at least one recording dir, or --all')

    counts = {'repaired': 0, 'skipped': 0, 'failed': 0}
    for d in dirs:
        try:
            status, detail = repair_session(d, dry_run=args.dry_run)
        except Exception as exc:  # one bad session must not abort a 300-session sweep
            status, detail = 'failed', f'{type(exc).__name__}: {exc}'
        counts[status] += 1
        if status != 'skipped':
            print(f'[{status}] {os.path.basename(d)}: {detail}')

    print(f'\nrepaired={counts["repaired"]} skipped={counts["skipped"]} failed={counts["failed"]}')
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
