"""
pipeline.py

Glue the root scripts share: resolving repo-relative paths in configs,
turning a config into farm_sim.launch.py arguments, writing the effective
control config for the controller node, and scoring a controller log.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from farmsim_utils.paths import CFG_DIR, REPO


def repo_path(p):
    """Absolute path for a config value: absolute stays, 'paths/x.yaml'
    resolves under cfg/ when it exists there, anything else under the repo."""
    if p is None:
        return None
    p = Path(p)
    if p.is_absolute():
        return p
    if (CFG_DIR / p).exists():
        return CFG_DIR / p
    return REPO / p


def timestamp():
    return time.strftime('%Y%m%d_%H%M%S')


def robot_config_file(variant):
    return REPO / 'robots' / 'thorvald_model' / 'config' / variant / f'{variant}.yaml'


def sim_launch_mapping(cfg, spawn, camera=None, world=None, **extra):
    """farm_sim.launch.py arguments from the merged robot/farm config.
    `spawn` = dict(x, y, yaw[, z]); `camera` overrides cfg robot.camera."""
    robot = cfg['robot']
    cam = dict(robot['camera'], **(camera or {}))
    mapping = {
        'field': cfg['farm']['field'],
        'world': str(world) if world else '',
        'robot_config': str(robot_config_file(robot['variant'])),
        'x': f"{spawn['x']:.3f}", 'y': f"{spawn['y']:.3f}",
        'z': spawn.get('z', robot.get('spawn_z', 2.0)), 'yaw': f"{spawn['yaw']:.4f}",
        'camera_height': cam['height'], 'camera_pitch': cam['pitch'], 'camera_x': cam['x'],
        'camera_roll': cam['roll'], 'camera_yaw': cam['yaw'], 'camera_hfov': cam['hfov'],
        'camera_noise_stddev': cam['noise_stddev'],
        'headless': cfg['sim'].get('headless', True),
    }
    mapping.update(extra)
    return mapping


def write_control_config(control_cfg, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = dict(control_cfg)
    if cfg.get('checkpoint'):
        cfg['checkpoint'] = str(repo_path(cfg['checkpoint']))
    with open(out_path, 'w') as f:
        yaml.safe_dump({'control': cfg}, f, sort_keys=False)
    return out_path


def read_log(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # last line may be half-written
    return rows


def wait_for_completion(log_path, max_duration_s, sim=None, poll_s=2.0, stall_s=60.0):
    """Block until the controller logs `done`, the run times out, the robot
    stalls (no progress for stall_s) or the sim dies. Returns the reason."""
    t0 = time.monotonic()
    best_s, best_t = -1.0, t0
    while True:
        rows = read_log(log_path)
        if rows and rows[-1].get('done'):
            return 'done'
        if rows and rows[-1]['s'] > best_s + 0.05:
            best_s, best_t = rows[-1]['s'], time.monotonic()
        now = time.monotonic()
        if now - t0 > max_duration_s:
            return 'timeout'
        if rows and now - best_t > stall_s:
            return 'stalled'
        if sim is not None and not sim.alive():
            return 'sim_exited'
        time.sleep(poll_s)


def score_log(rows):
    """Tracking metrics of one run. In-row error is the GROUND-TRUTH error,
    whatever pose source the controller used."""
    if not rows:
        return {'frames': 0}
    in_row = [r for r in rows if r['kind'] == 'row']
    e = np.array([r['e_gt'] for r in in_row]) if in_row else np.zeros(1)
    psi = np.array([r['psi_gt'] for r in in_row]) if in_row else np.zeros(1)
    t = np.array([r['t'] for r in rows])
    v = np.array([r['v'] for r in rows])
    est = [r for r in in_row if r.get('e_est') is not None]
    out = {
        'frames': len(rows),
        'completed': bool(rows[-1].get('done')),
        'progress': float(rows[-1]['s'] / max(rows[-1].get('total', 1.0), 1e-6)),
        'duration_s': float(t[-1] - t[0]),
        'mean_speed': float(v.mean()),
        'in_row_rmse_offset_m': float(math.sqrt((e ** 2).mean())),
        'in_row_max_offset_m': float(np.abs(e).max()),
        'in_row_rmse_heading_deg': float(np.degrees(math.sqrt((psi ** 2).mean()))),
    }
    if est:
        out['estimate_frames'] = len(est)
        out['estimate_mae_offset_m'] = float(np.mean([abs(r['e_est'] - r['e_gt']) for r in est]))
        out['estimate_mae_heading_deg'] = float(np.degrees(np.mean([abs(r['psi_est'] - r['psi_gt']) for r in est])))
        out['mean_confidence'] = float(np.mean([r['conf_est'] for r in est]))
    return out
