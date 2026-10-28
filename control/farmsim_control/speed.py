"""
speed.py

Shared linear-velocity policy. Strategies compute omega; unless a strategy
owns its speed (the fuzzy controller does), v comes from here:

    v = v_nominal
        / (1 + k_turn * |omega|)                       slow down in turns
        / (1 + k_heading * |psi| + k_offset * |e|)     slow down when badly aligned
        * confidence_scale                             trust the camera less -> drive slower
        * row_end_scale                                ramp down before the row end

then clipped to [v_min, v_max]. Written in torch so the same code runs in the
ROS node and inside the differentiable optimisation (optimize.py).
"""

import torch
import torch.nn as nn


class SpeedPolicy(nn.Module):

    def __init__(self, v_nominal=0.3, v_min=0.05, v_max=0.6, k_turn=0.5, k_heading=0.5,
                 k_offset=0.3, min_confidence=0.3, confidence_floor=0.3, row_end_ramp_m=2.0,
                 row_end_floor=0.3, learnable=False):
        super().__init__()
        gains = {'k_turn': k_turn, 'k_heading': k_heading, 'k_offset': k_offset}
        for name, value in gains.items():
            t = torch.tensor(float(value))
            if learnable:
                self.register_parameter(name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        self.v_nominal, self.v_min, self.v_max = v_nominal, v_min, v_max
        self.min_confidence, self.confidence_floor = min_confidence, confidence_floor
        self.row_end_ramp_m, self.row_end_floor = row_end_ramp_m, row_end_floor

    def forward(self, obs, omega):
        e, psi = obs['lateral_offset'], obs['heading_error']
        v = self.v_nominal / (1 + self.k_turn.abs() * omega.abs())
        v = v / (1 + self.k_heading.abs() * psi.abs() + self.k_offset.abs() * e.abs())
        conf = obs.get('confidence')
        if conf is not None:
            trust = ((conf - self.min_confidence) / (1 - self.min_confidence)).clamp(0, 1)
            v = v * (self.confidence_floor + (1 - self.confidence_floor) * trust)
        dist = obs.get('dist_to_end')
        if dist is not None and self.row_end_ramp_m > 0:
            v = v * (dist / self.row_end_ramp_m).clamp(self.row_end_floor, 1.0)
        return v.clamp(self.v_min, self.v_max)
