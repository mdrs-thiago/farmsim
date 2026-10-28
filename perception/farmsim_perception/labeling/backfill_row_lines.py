#!/usr/bin/env python3
"""Backfill row-line labels into already-exported session manifests.

Older sessions were exported before row_line_* fields existed.  Their raw
camera poses and calibration are still present, so re-recording or rewriting
videos/bags is unnecessary.  This tool uses the exporter's exact projection
code and atomically replaces only manifest.json.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path



from farmsim_perception.labeling.export_recording import (  # noqa: E402
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_PITCH,
    DEFAULT_CAMERA_ROLL,
    DEFAULT_CAMERA_X,
    DEFAULT_CAMERA_YAW,
    _compute_row_line,
    _compute_row_state,
)
from farmsim_farm.row_geometry import load_rows  # noqa: E402


LINE_KEYS = (
    'row_line_px', 'row_line_norm', 'row_line_in_frame',
    'row_line_angle_deg', 'row_line_lookahead_m',
)


def _atomic_json(path, value):
    fd, tmp = tempfile.mkstemp(prefix='.manifest.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def backfill_manifest(manifest_path, recording_dir, rows, write=False):
    manifest = json.loads(manifest_path.read_text())
    camera_path = recording_dir / 'camera_info.json'
    poses_path = recording_dir / 'poses.jsonl'
    params_path = recording_dir / 'run_params.json'
    missing = [p.name for p in (camera_path, poses_path, params_path) if not p.is_file()]
    if missing:
        return {'status': 'skipped', 'reason': 'missing ' + ', '.join(missing)}

    camera = json.loads(camera_path.read_text())
    params = json.loads(params_path.read_text())
    poses = {p['frame']: p for p in map(json.loads, poses_path.read_text().splitlines())}
    extrinsics = {
        'camera_height': float(params.get('camera_height', DEFAULT_CAMERA_HEIGHT)),
        'camera_pitch': float(params.get('camera_pitch', DEFAULT_CAMERA_PITCH)),
        'camera_x': float(params.get('camera_x', DEFAULT_CAMERA_X)),
        'camera_roll': float(params.get('camera_roll', DEFAULT_CAMERA_ROLL)),
        'camera_yaw': float(params.get('camera_yaw', DEFAULT_CAMERA_YAW)),
    }

    changed = projectable = near_visible = far_visible = both_visible = 0
    for frame in manifest.get('per_frame', []):
        pose = poses.get(frame.get('frame'))
        line = None
        if pose is not None:
            state = _compute_row_state(pose['T_world_optical'], rows, **extrinsics)
            if state['row_id'] is not None:
                line = _compute_row_line(
                    pose['T_world_optical'], rows, state['row_id'], camera,
                    state['base_pos'], state['base_yaw'])
        values = line or {key: None for key in LINE_KEYS}
        if any(key not in frame or frame.get(key) != values[key] for key in LINE_KEYS):
            changed += 1
        for key in LINE_KEYS:
            frame[key] = values[key]
        if line:
            projectable += 1
            near, far = line['row_line_in_frame']
            near_visible += int(near)
            far_visible += int(far)
            both_visible += int(near and far)

    if write and changed:
        _atomic_json(manifest_path, manifest)
    return {
        'status': 'updated' if write and changed else ('would-update' if changed else 'unchanged'),
        'frames': len(manifest.get('per_frame', [])), 'changed': changed,
        'projectable': projectable, 'near_visible': near_visible,
        'far_visible': far_visible, 'both_visible': both_visible,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', default='datasets/sim/vollebekk_farm_20250425__20250425')
    ap.add_argument('--recordings-root', default='outputs/recordings')
    ap.add_argument('--rows-file', default='farm/fields/20250425/rows_centerline.yaml')
    ap.add_argument('--session', action='append', help='limit to this session (repeatable)')
    ap.add_argument('--write', action='store_true', help='apply changes; default is a dry run')
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    recordings_root = Path(args.recordings_root)
    rows = load_rows(args.rows_file)
    wanted = set(args.session or [])
    manifests = sorted(dataset_root.glob('*/manifest.json'))
    if wanted:
        manifests = [p for p in manifests if p.parent.name in wanted]

    totals = {k: 0 for k in ('frames', 'changed', 'projectable', 'near_visible',
                              'far_visible', 'both_visible')}
    skipped = []
    for manifest_path in manifests:
        result = backfill_manifest(
            manifest_path, recordings_root / manifest_path.parent.name, rows, args.write)
        if result['status'] == 'skipped':
            skipped.append((manifest_path.parent.name, result['reason']))
            continue
        for key in totals:
            totals[key] += result[key]

    mode = 'WRITE' if args.write else 'DRY RUN'
    print(f'{mode}: {len(manifests)} session(s), {totals["frames"]} frames')
    print(f'  changed={totals["changed"]} projectable={totals["projectable"]}')
    print(f'  visible: near={totals["near_visible"]} far={totals["far_visible"]} '
          f'both={totals["both_visible"]}')
    print(f'  skipped={len(skipped)}')
    for session, reason in skipped:
        print(f'    {session}: {reason}')


if __name__ == '__main__':
    main()
