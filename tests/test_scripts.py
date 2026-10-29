"""Smoke tests for the root scripts. The simulator is not launched; the
offline paths (path selection, offline perception, control-aware
optimisation) run for real on one dataset sequence."""

import json
import subprocess
import sys

import pytest
import torch
import yaml

from farmsim_utils.paths import DEFAULT_DATASET, REPO

HAS_DATA = (DEFAULT_DATASET / 'splits' / 'test.txt').exists()


def run(*args):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


@pytest.mark.parametrize('script', ['select_path.py', 'run_perception.py', 'record_dataset.py',
                                    'run_control.py', 'optimize.py'])
def test_help(script):
    assert 'usage' in run(script, '--help')


def test_select_path_new_and_existing(tmp_path):
    out = tmp_path / 'p.yaml'
    text = run('select_path.py', '--field', '20250425', '--passes', '3:west:east', '5:east:west',
               '--out', out, '--no-preview')
    assert 'pass 1: row 5 east -> west' in text
    assert yaml.safe_load(out.read_text())['passes'][1] == {'row': 5, 'enter': 'east', 'exit': 'west'}
    assert 'preview' in run('select_path.py', 'cfg/paths/example.yaml')


def test_run_control_launch_arguments():
    from farmsim_farm.paths import FieldPath
    from farmsim_utils.config import load_yaml
    from farmsim_utils.pipeline import repo_path, sim_launch_mapping
    cfg = load_yaml('run_control.yaml')
    x, y, yaw = FieldPath.load(repo_path(cfg['run']['path'])).spawn_pose()
    m = sim_launch_mapping(cfg, {'x': x, 'y': y, 'yaw': yaw}, control=True)
    assert m['field'] == 20250425 and m['control'] is True
    assert m['robot_config'].endswith(f"{cfg['robot']['variant']}.yaml")


def test_record_session_controller_and_path(tmp_path):
    import record_dataset
    from farmsim_farm.fields import Field
    from farmsim_farm.paths import FieldPath
    from farmsim_utils.config import load_yaml
    cfg = load_yaml('record_dataset.yaml')
    params = {'control_strategy': 'stanley', 'k_offset': 0.3, 'k_heading': 0.5, 'lookahead_m': 1.2,
              'k_stanley': 0.7, 'stanley_v_epsilon': 0.5, 'stanley_k_damping': 0.6,
              'bangbang_deadband_m': 0.1, 'bangbang_magnitude': 0.2, 'forward_speed': 0.25,
              'perturb_std': 0.02, 'seed': 1, 'row_id': 4, 'spawn': {'x': 30.0, 'y': -6.1, 'yaw': 3.1}}
    c = record_dataset.session_control(cfg, params)
    assert c['strategy'] == 'stanley' and c['speed']['v_nominal'] == 0.25
    p = record_dataset.session_path(Field('20250425'), params, tmp_path / 'path.yaml')
    fp = FieldPath.load(p)
    seg = [s for s in fp.reference() if s.kind == 'row'][0]
    assert seg.b[0] < seg.a[0]   # yaw ~ pi: driving toward -x


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    from farmsim_perception.models.row_net import make_model
    path = tmp_path_factory.mktemp('ckpt') / 'last.pt'
    torch.save({'model': 'cnn', 'state_dict': make_model('cnn', pretrained=False).state_dict(),
                'epoch': 0, 'args': {}}, path)
    return path


@pytest.mark.skipif(not HAS_DATA, reason='dataset/row_perception_v1 not present')
def test_run_perception_offline(checkpoint):
    text = run('run_perception.py', f'perception.checkpoint={checkpoint}', 'offline.limit=1',
               'offline.video=false', 'perception.device=cpu')
    assert 'mean' in text


@pytest.mark.skipif(not HAS_DATA, reason='dataset/row_perception_v1 not present')
def test_optimize_control_aware_one_epoch(checkpoint):
    text = run('optimize.py', 'optimize.name=pytest_smoke', f'optimize.control_aware.checkpoint={checkpoint}',
               'optimize.control_aware.epochs=1', 'optimize.control_aware.limit_sequences=1',
               'optimize.control_aware.batch_size=8', 'optimize.control_aware.closed_loop_eval=false')
    out = REPO / 'models' / 'optimize' / 'pytest_smoke'
    hist = json.loads((out / 'history.json').read_text())
    assert len(hist) == 1 and (out / 'controller.pt').exists() and 'saved to' in text
    trained = yaml.safe_load((out / 'control.yaml').read_text())['control']
    from farmsim_control.laws import make_controller
    make_controller(trained)   # the saved controller loads back
    import shutil
    shutil.rmtree(out)
