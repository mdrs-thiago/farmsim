"""
fuzzy.py

Fuzzy row-following controller built on torch-fis (`fuzzy_torch`). Two
first-order TSK systems:

  omega-FIS  inputs: lateral offset e, heading error psi, yaw rate r
             9 rules over e x psi (Neg / Zero / Pos each), e.g.
               IF e is Pos (robot left of the row) AND psi is Zero THEN turn clockwise
             A rule's constant term is -k_offset*e_c + k_heading*psi_c at its
             term centres, so the untrained rule base is a smoothly
             interpolated PD law; every rule also carries -k_damping*r.
             Training then specialises each region.

  v-FIS      inputs: |psi|, |e|, perception confidence, distance to row end
             16 zero-order rules (Small/Large, Small/Large, Low/High,
             Near/Far). Each rule's speed is v_nominal times one factor per
             "bad" condition: large heading error, large offset, low
             confidence, near the row end.

Every membership function centre/width and every consequent coefficient is
an nn.Parameter, so optimize.py can train the whole controller jointly with
the CNN. Inputs are clamped to each variable's range before inference.
"""

import itertools

import torch

from fuzzy_torch import FuzzyRule, FuzzyVariable, GaussianMF, TSKConsequent, TSKSystem

from farmsim_control.laws import Controller


def _three_terms(name, half_range, sigma_frac=0.45):
    var = FuzzyVariable(name, -half_range, half_range)
    s = sigma_frac * half_range
    var.add_term('Neg', GaussianMF(-half_range, s))
    var.add_term('Zero', GaussianMF(0.0, s))
    var.add_term('Pos', GaussianMF(half_range, s))
    return var


def _two_terms(name, lo, hi, low_name, high_name, sigma_frac=0.45):
    var = FuzzyVariable(name, lo, hi)
    s = sigma_frac * (hi - lo)
    var.add_term(low_name, GaussianMF(lo, s))
    var.add_term(high_name, GaussianMF(hi, s))
    return var


class FuzzyController(Controller):
    owns_speed = True

    def __init__(self, e_range=1.0, psi_range=0.8, r_range=0.6, end_range=10.0,
                 k_offset=0.3, k_heading=0.5, k_damping=0.4, v_nominal=None,
                 slow_heading=0.5, slow_offset=0.6, slow_confidence=0.4, slow_row_end=0.4,
                 **kw):
        super().__init__(**kw)
        self.ranges = {'E': e_range, 'Psi': psi_range, 'R': r_range}
        v_nominal = self.speed.v_nominal if v_nominal is None else v_nominal

        # omega-FIS
        e, psi, r = _three_terms('E', e_range), _three_terms('Psi', psi_range), _three_terms('R', r_range)
        out = FuzzyVariable('Omega', -self.max_angular, self.max_angular)
        centres = {'Neg': -1.0, 'Zero': 0.0, 'Pos': 1.0}
        rules = []
        for te, tp in itertools.product(centres, centres):
            c0 = -k_offset * centres[te] * e_range + k_heading * centres[tp] * psi_range
            # order-1 coefficients act on inputs normalised to [-1, 1]: [1, e, psi, r]
            coeffs = [c0, 0.0, 0.0, -k_damping * r_range]
            rules.append(FuzzyRule([(e, te), (psi, tp)], TSKConsequent(order=1, coefficients=coeffs)))
        self.omega_fis = TSKSystem(inputs=[e, psi, r], output=out, rules=rules)

        # v-FIS
        abs_psi = _two_terms('AbsPsi', 0.0, psi_range, 'Small', 'Large')
        abs_e = _two_terms('AbsE', 0.0, e_range, 'Small', 'Large')
        conf = _two_terms('Conf', 0.0, 1.0, 'Low', 'High')
        end = _two_terms('End', 0.0, end_range, 'Near', 'Far')
        out_v = FuzzyVariable('V', 0.0, self.speed.v_max)
        bad = {('AbsPsi', 'Large'): slow_heading, ('AbsE', 'Large'): slow_offset,
               ('Conf', 'Low'): slow_confidence, ('End', 'Near'): slow_row_end}
        rules = []
        for combo in itertools.product(*[[(v, t) for t in v.terms] for v in (abs_psi, abs_e, conf, end)]):
            speed = v_nominal
            for var, term in combo:
                speed *= bad.get((var.name, term), 1.0)
            rules.append(FuzzyRule(list(combo), TSKConsequent(order=0, coefficients=[speed])))
        self.v_fis = TSKSystem(inputs=[abs_psi, abs_e, conf, end], output=out_v, rules=rules)
        self.end_range = end_range

    def forward(self, obs):
        e = obs['lateral_offset'].clamp(-self.ranges['E'], self.ranges['E'])
        psi = obs['heading_error'].clamp(-self.ranges['Psi'], self.ranges['Psi'])
        r = obs['yaw_rate'].clamp(-self.ranges['R'], self.ranges['R'])
        omega = self.omega_fis({'E': e, 'Psi': psi, 'R': r}).clamp(-self.max_angular, self.max_angular)
        conf = obs.get('confidence', torch.ones_like(e))
        end = obs.get('dist_to_end', torch.full_like(e, self.end_range)).clamp(0, self.end_range)
        v = self.v_fis({'AbsPsi': psi.abs(), 'AbsE': e.abs(), 'Conf': conf.clamp(0, 1), 'End': end})
        return v.clamp(self.speed.v_min, self.speed.v_max), omega
