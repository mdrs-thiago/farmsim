#!/usr/bin/env python3
"""
record_dataset.py -- record labelled sequences for training the perception.

Per session:
  1. randomise the world (lighting, weeds) and the run (spawn row and jitter,
     controller strategy and gains, camera mount) from one seed
  2. launch the sim; the controller drives the spawn row (ground truth by
     default, or the CNN for DAgger-style collection: record.pose_source)
  3. record camera frames with the ground-truth camera pose until
     target_frames
  4. render the vegetation/structure labels from those exact poses
     (label mesh + rasters of the field) and, optionally, a coarse
     alignment check
  5. export and add the session to the dataset in the row_perception_v1
     layout (images, masks, annotations.jsonl, meta.json), assign it a
     split (by world seed) and append it to sessions.csv

  python record_dataset.py record.sessions=20
  python record_dataset.py record.seed_start=900001 record.sessions=5 sim.headless=false
  python record_dataset.py record.pose_source=perception record.checkpoint=models/row_net/cnn/last.pt

Raw recordings stay in outputs/recordings/<session>/ (frames, poses,
logs); the bag is only kept with record.keep_bag=true.
"""

import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time

import yaml

from farmsim_farm import world_gen
from farmsim_farm.fields import Field
from farmsim_farm.paths import load_rows_by_id
from farmsim_utils.config import parse
from farmsim_utils.paths import GENERATED_WORLDS_DIR, OUTPUTS_DIR, RECORDINGS_DIR
from farmsim_utils.pipeline import repo_path, sim_launch_mapping, write_control_config
from farmsim_utils.sim import Sim

SPLIT_FRACTIONS = {'test': 0.15, 'val': 0.15}


def session_control(cfg, params):
    """The recording's controller: the strategy and gains world_gen drew,
    with the old dataset's speed law (v = v_nom / (1 + 0.5 |omega|))."""
    c = json.loads(json.dumps(cfg['control']))
    c['strategy'] = params['control_strategy']
    c['gains'] = {
        'pd': {'k_offset': params['k_offset'], 'k_heading': params['k_heading'], 'k_damping': 0.4},
        'pure_pursuit': {'lookahead_m': params['lookahead_m']},
        'stanley': {'k_stanley': params['k_stanley'], 'v_epsilon': params['stanley_v_epsilon'],
                    'k_damping': params['stanley_k_damping']},
        'bangbang': {'deadband_m': params['bangbang_deadband_m'], 'magnitude': params['bangbang_magnitude']},
    }
    c['speed'] = dict(c.get('speed', {}), v_nominal=params['forward_speed'], k_turn=0.5,
                      k_heading=0.0, k_offset=0.0, row_end_ramp_m=0.0)
    c['runtime'] = dict(c.get('runtime', {}), perturb_std=params['perturb_std'], seed=params['seed'])
    return c


def session_path(field, params, out_file):
    """Single pass along the spawn row, in the direction the robot faces."""
    import math
    rows = load_rows_by_id(field.rows_file)
    row_id = params['row_id']
    pts = rows[row_id]
    d = pts[-1] - pts[0]
    yaw = params['spawn']['yaw']
    forward = d[0] * math.cos(yaw) + d[1] * math.sin(yaw) > 0
    path = {'field': field.date, 'lead_in_m': 0.0, 'lead_out_m': 2.0,
            'passes': [{'row': int(row_id), 'enter': 'start' if forward else 'end',
                        'exit': 'end' if forward else 'start'}]}
    with open(out_file, 'w') as f:
        yaml.safe_dump(path, f, sort_keys=False)
    return out_file


def count_frames(rec_dir):
    p = rec_dir / 'poses.jsonl'
    return sum(1 for _ in open(p)) if p.exists() else 0


def py(module, *args, log=None):
    cmd = [sys.executable, '-m', module, *map(str, args)]
    with open(log, 'w') if log else open(os.devnull, 'w') as out:
        subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.STDOUT)


def add_to_dataset(dataset, session, seed, params, stats):
    from farmsim_perception.labeling.build_dataset import stable_score
    splits = {s: set((dataset / 'splits' / f'{s}.txt').read_text().split())
              if (dataset / 'splits' / f'{s}.txt').exists() else set() for s in ('train', 'val', 'test')}
    # sessions of the same world seed share a split
    split = next((s for s, names in splits.items() if any(n.endswith(f'seed{seed}') for n in names)), None)
    if split is None:
        score = stable_score(f'row_perception_v1:{seed}')
        split = 'test' if score < SPLIT_FRACTIONS['test'] else (
            'val' if score < SPLIT_FRACTIONS['test'] + SPLIT_FRACTIONS['val'] else 'train')
    (dataset / 'splits').mkdir(parents=True, exist_ok=True)
    if session not in splits[split]:
        with open(dataset / 'splits' / f'{split}.txt', 'a') as f:
            f.write(session + '\n')
    csv_path = dataset / 'sessions.csv'
    new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['session', 'split', 'seed', 'frames', 'strategy', 'measured_quality', 'rgb_source'])
        w.writerow([session, split, seed, stats['frames'], params['control_strategy'],
                    params.get('trajectory_quality', '?'), stats['rgb_source']])
    return split


def record_session(cfg, seed):
    rc = cfg['record']
    field = Field(cfg['farm']['field'])
    if not field.has_ground_truth:
        raise SystemExit(f'field {field.date} has no ground-truth assets ({field.missing_ground_truth}); '
                         'build them with farm/tools first')
    session = f'run_{time.strftime("%Y%m%d_%H%M%S")}_seed{seed}'
    rec = RECORDINGS_DIR / session
    rec.mkdir(parents=True, exist_ok=True)
    print(f'=== {session}')

    # 1. randomise
    args = world_gen.default_args(
        rows_file=str(field.rows_file), world_template=str(field.world),
        out_world_dir=str(GENERATED_WORLDS_DIR), session_name=session, seed=seed,
        navigation_profile=rc['navigation_profile'], terrain_heightmap=str(field.heightmap),
        spawn_z=cfg['robot'].get('spawn_z'))
    params = world_gen.generate(args)
    with open(rec / 'run_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    control_cfg = write_control_config(session_control(cfg, params), rec / 'control.yaml')
    path_file = session_path(field, params, rec / 'path.yaml')
    camera = {'height': params['camera_height'], 'pitch': params['camera_pitch'], 'x': params['camera_x'],
              'roll': params['camera_roll'], 'yaw': params['camera_yaw'], 'hfov': params['camera_hfov'],
              'noise_stddev': params['camera_noise_stddev']}
    perception = rc['pose_source'] == 'perception'
    mapping = sim_launch_mapping(
        cfg, params['spawn'], camera=camera, world=params['world_path'],
        control=True, control_config=control_cfg, path_file=path_file, pose_source=rc['pose_source'],
        control_log=rec / 'control_log.jsonl', perception=perception,
        checkpoint=repo_path(rc['checkpoint']) if perception else '')

    # 2-3. drive and record
    with Sim(mapping, str(rec / 'sim.log'), startup_wait_s=cfg['sim'].get('startup_wait_s', 12)) as sim:
        recorder = sim.run([
            'ros2', 'run', 'farmsim_perception', 'record_camera_poses', '--ros-args',
            '-p', f'world_name:={field.world_name}', '-p', f'output_dir:={rec}',
            '-p', f'save_every_n:={rc["save_every_n"]}',
            '-p', f'camera_height:={camera["height"]}', '-p', f'camera_pitch:={camera["pitch"]}',
            '-p', f'camera_x_offset:={camera["x"]}', '-p', f'camera_roll:={camera["roll"]}',
            '-p', f'camera_yaw:={camera["yaw"]}'], log_path=str(rec / 'record.log'))
        t0 = time.monotonic()
        while count_frames(rec) < rc['target_frames']:
            if time.monotonic() - t0 > rc['max_record_s']:
                print(f'  hit the {rc["max_record_s"]} s cap at {count_frames(rec)} frames')
                break
            if recorder.poll() is not None:
                print('  recorder exited early, see record.log')
                break
            time.sleep(2)
        recorder.send_signal(2)
        try:
            recorder.wait(timeout=15)
        except subprocess.TimeoutExpired:
            recorder.kill()
    n = count_frames(rec)
    print(f'  {n} frames in {time.monotonic() - t0:.0f} s')
    if n < 5:
        raise RuntimeError(f'only {n} frames recorded, see {rec}/sim.log and record.log')

    # 4. labels
    py('farmsim_perception.labeling.render_labels', '--mesh', field.label_mesh,
       '--vegetation-raster', field.vegetation_raster, '--structure-raster', field.structure_raster,
       '--poses', rec / 'poses.jsonl', '--camera-info', rec / 'camera_info.json',
       '--out-dir', rec / 'label_frames', log=rec / 'render.log')
    if rc.get('alignment_check') and not field.topdown_image.is_file():
        print('  skipping alignment check: no topdown.png (python -m farmsim_utils.fetch_assets fetch)')
    elif rc.get('alignment_check'):
        py('farmsim_perception.labeling.check_alignment', '--recording-dir', rec, '--georef', field.georef,
           '--mesh', field.label_mesh, '--topdown-meta', field.topdown_meta_file, '--resolution', 0.1,
           '--limit', 10, '--out-dir', rec / 'alignment_check', log=rec / 'alignment_check.log')

    # 5. export -> dataset
    export_root = OUTPUTS_DIR / 'sim_export'
    py('farmsim_perception.labeling.export_recording', '--recording-dir', rec, '--world-name', field.world_name,
       '--survey-date', field.date, '--session-name', session, '--rows-file', field.rows_file,
       '--out-root', export_root, log=rec / 'export.log')
    export_dir = export_root / f'{field.world_name}__{field.date}' / session
    from farmsim_perception.labeling.build_dataset import build_session
    dataset = repo_path(rc['dataset'])
    _, stats = build_session((str(export_dir), str(dataset), True, None))
    split = add_to_dataset(dataset, session, seed, params, stats)
    if not rc.get('keep_bag'):
        shutil.rmtree(export_dir, ignore_errors=True)
    print(f'  added to {dataset.name} ({split}, {stats["frames"]} frames)')
    return session


def main():
    cfg, _ = parse(__doc__, 'record_dataset.yaml')
    rc = cfg['record']
    rng = random.Random()
    done, failed = [], []
    for i in range(rc['sessions']):
        seed = rc['seed_start'] + i if rc.get('seed_start') is not None else rng.randrange(10**9)
        try:
            done.append(record_session(cfg, seed))
        except Exception as exc:  # one bad session must not end an overnight campaign
            print(f'  FAILED seed={seed}: {exc}')
            failed.append(seed)
        with open(OUTPUTS_DIR / 'record_campaign.jsonl', 'a') as f:
            f.write(json.dumps({'seed': seed, 'ok': seed not in failed, 'time': time.time()}) + '\n')
    print(f'{len(done)} session(s) recorded, {len(failed)} failed {failed or ""}')
    return 1 if failed and not done else 0


if __name__ == '__main__':
    sys.exit(main())
