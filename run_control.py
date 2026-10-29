#!/usr/bin/env python3
"""
run_control.py -- drive a path in the simulator with the full stack.

pose_source: perception (default) runs the CNN on the camera and the
controller on its estimate (CNN + control); pose_source: gt is the
ground-truth baseline. Either way the score uses the ground-truth error.

  python run_control.py                                    # cfg/run_control.yaml
  python run_control.py control.strategy=fuzzy run.path=paths/single_row.yaml
  python run_control.py run.pose_source=gt run.repeats=3 sim.headless=false
  python run_control.py run.pose_source=gt run.video=true run.max_duration_s=120

Each run writes outputs/runs/<name>/ with the effective config, the
controller log (control_log.jsonl), sim.log and metrics.json. run.video=true
adds a chase camera to the robot and records chase_camera.mp4,
front_camera.mp4 and video.mp4 (both cameras with the controller state
drawn on); `python -m farmsim_utils.video compose` puts runs side by side.
"""

import json
import signal
import subprocess
import sys

import yaml

from farmsim_farm.paths import FieldPath
from farmsim_utils.config import parse
from farmsim_utils.paths import RUNS_DIR
from farmsim_utils.pipeline import (repo_path, score_log, read_log, sim_launch_mapping, timestamp,
                                    wait_for_completion, write_control_config)
from farmsim_utils.sim import Sim


def run(cfg, out_dir):
    """One closed-loop run; returns the metrics dict. Used by optimize.py too."""
    run_cfg = cfg['run']
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    path_file = repo_path(run_cfg['path'])
    path = FieldPath.load(path_file)
    if str(path.field) != str(cfg['farm']['field']):
        cfg['farm']['field'] = path.field
    x, y, yaw = path.spawn_pose()
    # perception_node runs whenever the controller uses it; run.perception=true
    # also runs it next to a ground-truth controller (run_perception.py --live).
    perception = run_cfg['pose_source'] == 'perception' or bool(run_cfg.get('perception'))
    control_cfg = write_control_config(cfg['control'], out_dir / 'control.yaml')
    log = out_dir / 'control_log.jsonl'
    mapping = sim_launch_mapping(
        cfg, {'x': x, 'y': y, 'yaw': yaw},
        control=True, control_config=control_cfg, path_file=path_file,
        pose_source=run_cfg['pose_source'], control_log=log,
        perception=perception,
        checkpoint=repo_path(run_cfg['checkpoint']) if perception else '',
        perception_view=perception and not cfg['sim'].get('headless', True))
    video = bool(run_cfg.get('video'))
    if video:
        mapping['model_extras'] = repo_path('robots/thorvald_model/urdf/sensor_modules/gz_camera_chase.xacro')
    if perception and not (run_cfg.get('checkpoint') and repo_path(run_cfg['checkpoint']).exists()):
        raise SystemExit(f"checkpoint {run_cfg['checkpoint']} not found; train one first "
                         '(python -m farmsim_perception.training.train) or set run.pose_source=gt')
    with Sim(mapping, str(out_dir / 'sim.log'), startup_wait_s=cfg['sim'].get('startup_wait_s', 12)) as sim:
        recorder = sim.run([sys.executable, '-m', 'farmsim_utils.video', 'record', str(out_dir)],
                           log_path=str(out_dir / 'video.log')) if video else None
        try:
            reason = wait_for_completion(log, run_cfg['max_duration_s'], sim)
        finally:
            if recorder:
                recorder.send_signal(signal.SIGTERM)
                try:
                    recorder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    recorder.kill()
    if video:
        from farmsim_utils.video import compose
        compose(out_dir / 'video.mp4', [out_dir])
    metrics = score_log(read_log(log))
    metrics['end_reason'] = reason
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    cfg, _ = parse(__doc__, 'run_control.yaml')
    name = cfg['run'].get('name') or f"{timestamp()}_{cfg['control']['strategy']}_{cfg['run']['pose_source']}"
    results = []
    for i in range(cfg['run'].get('repeats', 1)):
        out = RUNS_DIR / (name if cfg['run'].get('repeats', 1) == 1 else f'{name}/rep{i}')
        m = run(cfg, out)
        results.append(m)
        print(f'[{out.name}] ' + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                                          for k, v in m.items()))
    return 0 if all(r.get('completed') for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
