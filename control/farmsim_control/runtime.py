"""
runtime.py

Pieces of the control loop that only make sense on the running robot, not in
the differentiable optimisation: the command filter (clip, angular rate
limit, linear acceleration limit, optional exploration noise) and the
headland pure-pursuit used between row passes.
"""

import math

import numpy as np

from farmsim_farm.row_geometry import normalize_angle


class CommandFilter:

    def __init__(self, max_angular=0.5, max_angular_rate=0.6, max_linear_accel=0.5,
                 perturb_std=0.0, seed=None):
        self.max_angular = max_angular
        self.max_angular_rate = max_angular_rate
        self.max_linear_accel = max_linear_accel
        self.perturb_std = perturb_std
        self.rng = np.random.default_rng(seed)
        self.v = 0.0
        self.w = 0.0

    def __call__(self, v, w, dt):
        dt = dt if 0.0 < dt <= 0.5 else 0.05
        if self.perturb_std > 0:
            w += float(self.rng.normal(0.0, self.perturb_std))
        w = max(-self.max_angular, min(self.max_angular, w))
        dw = self.max_angular_rate * dt
        self.w += max(-dw, min(dw, w - self.w))
        dv = self.max_linear_accel * dt
        self.v += max(-dv, min(dv, v - self.v))
        return self.v, self.w


def headland_command(x, y, yaw, target, speed, lookahead_m, max_angular):
    """Pure pursuit toward `target` (a point on the reference polyline)."""
    alpha = normalize_angle(math.atan2(target[1] - y, target[0] - x) - yaw)
    w = 2.0 * math.sin(alpha) / max(lookahead_m, 1e-3) * speed
    # Turn in place-ish when the target is behind: slow down, keep turning.
    v = speed * max(0.2, math.cos(alpha))
    return v, max(-max_angular, min(max_angular, w))
