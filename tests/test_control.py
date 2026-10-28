import math

import pytest
import torch

from farmsim_control.laws import BangBang, ControlState, make_controller
from farmsim_control.runtime import CommandFilter
from farmsim_utils.config import load_yaml

CONTROL = load_yaml('control.yaml')['control']
STRATEGIES = ['pd', 'pure_pursuit', 'stanley', 'bangbang', 'fuzzy']


def controller(strategy, **overrides):
    return make_controller(dict(CONTROL, strategy=strategy, **overrides)).eval()


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_left_offset_steers_clockwise(strategy):
    # robot 0.4 m left of the row, aligned: every strategy must turn right (omega < 0)
    v, w = controller(strategy).step(ControlState(0.4, 0.0))
    assert w < 0
    assert v > 0


@pytest.mark.parametrize('strategy', ['pd', 'pure_pursuit', 'stanley', 'fuzzy'])
def test_heading_error_steers_toward_row(strategy):
    _, w = controller(strategy).step(ControlState(0.0, 0.3))
    assert w > 0


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_commands_respect_limits(strategy):
    c = controller(strategy)
    v, w = c.step(ControlState(5.0, 2.0, yaw_rate=3.0))
    assert abs(w) <= CONTROL['max_angular'] + 1e-6
    assert CONTROL['speed']['v_min'] - 1e-6 <= v <= CONTROL['speed']['v_max'] + 1e-6


def test_speed_drops_with_low_confidence_and_row_end():
    c = controller('pd')
    v_ok, _ = c.step(ControlState(0.0, 0.0))
    v_blind, _ = c.step(ControlState(0.0, 0.0, confidence=0.1))
    v_end, _ = c.step(ControlState(0.0, 0.0, dist_to_end=0.2))
    assert v_blind < v_ok and v_end < v_ok


def test_fuzzy_speed_drops_with_low_confidence():
    c = controller('fuzzy')
    v_ok, _ = c.step(ControlState(0.0, 0.0, confidence=1.0))
    v_blind, _ = c.step(ControlState(0.0, 0.0, confidence=0.0))
    assert v_blind < 0.7 * v_ok


@pytest.mark.parametrize('strategy', ['pd', 'pure_pursuit', 'stanley', 'fuzzy'])
def test_gradients_reach_controller_and_inputs(strategy):
    c = controller(strategy).train()
    e = torch.tensor([0.2, -0.3], requires_grad=True)
    obs = {'lateral_offset': e, 'heading_error': torch.tensor([0.1, 0.0]),
           'yaw_rate': torch.zeros(2), 'speed': torch.zeros(2), 'confidence': torch.ones(2)}
    v, w = c(obs)
    (v.sum() + w.sum()).backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()
    assert any(p.grad is not None for p in c.parameters())


def test_bangbang_hysteresis_holds_side_inside_deadband():
    c = controller('bangbang')
    assert isinstance(c, BangBang)
    _, w1 = c.step(ControlState(0.5, 0.0))
    _, w2 = c.step(ControlState(0.0, 0.0))   # inside the deadband: keep turning the same way
    assert w1 < 0 and w2 == pytest.approx(w1)


def test_controller_checkpoint_roundtrip(tmp_path):
    c = controller('fuzzy')
    with torch.no_grad():
        for p in c.parameters():
            p.add_(0.01)
    torch.save(c.state_dict(), tmp_path / 'c.pt')
    c2 = controller('fuzzy', checkpoint=str(tmp_path / 'c.pt'))
    s = ControlState(0.3, -0.1, 0.05)
    assert c.step(s) == pytest.approx(c2.step(s))


def test_command_filter_rate_limits():
    f = CommandFilter(max_angular=0.5, max_angular_rate=0.6, max_linear_accel=0.5)
    v, w = f(1.0, 0.5, 0.1)
    assert v == pytest.approx(0.05) and w == pytest.approx(0.06)
    assert math.isfinite(f(0.0, 0.0, -1)[0])  # bad dt falls back to the default
