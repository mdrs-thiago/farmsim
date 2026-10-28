"""
laws.py

In-row steering strategies. Each one is an nn.Module mapping a batch of row
states to (v, omega), so the same object drives the robot from the ROS node
(`step`, one float state at a time) and trains by backprop in optimize.py
(`forward`, batched tensors). Gains are nn.Parameters.

Sign conventions (farmsim_farm.row_geometry):
  lateral_offset  e   > 0 when the robot is LEFT of the row's direction of travel
  heading_error   psi > 0 when the robot must turn counter-clockwise to align
  omega               > 0 counter-clockwise
so a left offset needs a negative omega and a positive psi a positive omega.

The strategies, deliberately different in structure (they were chosen for
trajectory diversity in the dataset, not to find the best controller):
  pd            omega = -k_offset*e + k_heading*psi - k_damping*r
  pure_pursuit  steer toward a point lookahead_m ahead on the (locally
                straight) row: alpha = psi + atan2(-e, L), omega = 2 sin(alpha) v / L
  stanley       omega = psi - atan2(k_stanley*e, v + v_eps) - k_damping*r
  bangbang      +/- magnitude toward the row with a hysteresis deadband
  fuzzy         TSK fuzzy system (fuzzy.py), also decides v

The yaw-rate damping terms are needed: commands reach the wheels through
several hops (controller -> twist_to_joints -> joint splitter -> gz bridge),
and the bare P / textbook Stanley laws oscillated and diverged against that
latency in recorded runs.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from farmsim_control.speed import SpeedPolicy


@dataclass
class ControlState:
    lateral_offset: float
    heading_error: float
    yaw_rate: float = 0.0
    speed: float = 0.0
    dist_to_end: float = math.inf
    confidence: float = 1.0

    def as_batch(self, device='cpu'):
        return {k: torch.tensor([float(min(v, 1e6))], device=device) for k, v in asdict(self).items()}


def _param(value):
    return nn.Parameter(torch.tensor(float(value)))


class Controller(nn.Module):
    owns_speed = False
    differentiable = True

    def __init__(self, speed=None, max_angular=0.5):
        super().__init__()
        self.speed = speed if speed is not None else SpeedPolicy()
        self.max_angular = max_angular

    def steer(self, obs):
        raise NotImplementedError

    def forward(self, obs):
        omega = self.steer(obs).clamp(-self.max_angular, self.max_angular)
        return self.speed(obs, omega), omega

    def reset(self):
        """Clear internal state between runs."""

    @torch.no_grad()
    def step(self, state: ControlState):
        v, omega = self(state.as_batch())
        return float(v[0]), float(omega[0])


class PD(Controller):
    def __init__(self, k_offset=0.3, k_heading=0.5, k_damping=0.4, **kw):
        super().__init__(**kw)
        self.k_offset, self.k_heading, self.k_damping = _param(k_offset), _param(k_heading), _param(k_damping)

    def steer(self, obs):
        return (-self.k_offset * obs['lateral_offset'] + self.k_heading * obs['heading_error']
                - self.k_damping * obs['yaw_rate'])


class PurePursuit(Controller):
    def __init__(self, lookahead_m=1.5, **kw):
        super().__init__(**kw)
        self.lookahead_m = _param(lookahead_m)

    def steer(self, obs):
        L = self.lookahead_m.abs() + 1e-3
        alpha = obs['heading_error'] + torch.atan2(-obs['lateral_offset'], L)
        return 2.0 * torch.sin(alpha) / L * self.speed.v_nominal


class Stanley(Controller):
    def __init__(self, k_stanley=0.8, v_epsilon=0.4, k_damping=0.6, **kw):
        super().__init__(**kw)
        self.k_stanley, self.v_epsilon, self.k_damping = _param(k_stanley), _param(v_epsilon), _param(k_damping)

    def steer(self, obs):
        v = self.speed.v_nominal + self.v_epsilon.abs()
        return (obs['heading_error'] - torch.atan2(self.k_stanley * obs['lateral_offset'], torch.as_tensor(v))
                - self.k_damping * obs['yaw_rate'])


class BangBang(Controller):
    """Relay with hysteresis. Inside the deadband the previous side is held,
    which stops chattering at e = 0. The batched/training path uses a tanh
    surrogate instead, since the relay has no useful gradient."""
    differentiable = False

    def __init__(self, deadband_m=0.1, magnitude=0.15, **kw):
        super().__init__(**kw)
        self.deadband_m, self.magnitude = _param(deadband_m), _param(magnitude)
        self._side = 0.0

    def reset(self):
        self._side = 0.0

    def steer(self, obs):
        e = obs['lateral_offset']
        if self.training or e.numel() > 1:
            return -self.magnitude * torch.tanh(3.0 * e / self.deadband_m.abs().clamp(min=1e-3))
        x = float(e[0])
        if x > float(self.deadband_m):
            self._side = -1.0
        elif x < -float(self.deadband_m):
            self._side = 1.0
        return self.magnitude * self._side * torch.ones_like(e)


STRATEGIES = {'pd': PD, 'pure_pursuit': PurePursuit, 'stanley': Stanley, 'bangbang': BangBang}


def make_controller(cfg):
    """cfg: the `control:` block of cfg/control.yaml -- strategy, gains,
    speed, fuzzy, max_angular and an optional `checkpoint` (state_dict saved
    by optimize.py) loaded on top of the configured initial values."""
    strategy = cfg.get('strategy', 'pd')
    speed = SpeedPolicy(**cfg.get('speed', {}))
    common = {'speed': speed, 'max_angular': cfg.get('max_angular', 0.5)}
    if strategy == 'fuzzy':
        from farmsim_control.fuzzy import FuzzyController
        controller = FuzzyController(**cfg.get('fuzzy', {}), **common)
    elif strategy in STRATEGIES:
        controller = STRATEGIES[strategy](**cfg.get('gains', {}).get(strategy, {}), **common)
    else:
        raise ValueError(f'unknown strategy {strategy!r}; choose from {sorted(STRATEGIES) + ["fuzzy"]}')
    if cfg.get('checkpoint'):
        controller.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu'))
    return controller
