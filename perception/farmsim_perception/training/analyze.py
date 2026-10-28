#!/usr/bin/env python3
"""
analyze.py

Is the CNN / CNN-LSTM difference real, and does memory help when frames are
actually unreliable?

1. Paired per-sequence statistics. Every test sequence is scored for both models
   under the same condition; the per-sequence differences are bootstrapped (10k
   resamples over sequences) for a 95% CI, and we count on how many sequences
   each model wins. A difference whose CI spans 0 is not evidence either way.

2. Stress conditions that single frames cannot resolve but a memory can:
     dropout_p   each frame is replaced by a flat grey frame with probability p
     burst_k     occluder covering 40-60% of the image that persists for k frames
                 at random places in the sequence (~30% of frames affected)
     noise_s     gaussian noise with std s on every frame
   Metrics are computed on ALL frames (the corrupted frames are where memory
   should matter) and separately on the corrupted frames only.

Writes runs/analysis_<split>.json and runs/analysis_<split>.md.

Usage:
  python -m farmsim_perception.training.analyze [--runs runs] [--models cnn cnn_lstm]
"""

import argparse
import json
import os

import numpy as np
import torch

from farmsim_perception.data import Sequences
from farmsim_perception.engine import Metrics, seq_generator
from farmsim_perception.training.evaluate import load
from farmsim_perception.models.row_net import RowCNNLSTM

from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401
KEY_METRICS = ['seg_mIoU', 'seg_flicker_pct', 'det_F1@0.5', 'det_mean_IoU', 'line_far_err_px',
               'lateral_MAE_cm', 'heading_MAE_deg', 'lateral_jitter_cm']
LOWER_IS_BETTER = ('err', 'MAE', 'jitter', 'flicker')
CONDITIONS = ['clean', 'dropout_0.1', 'dropout_0.3', 'burst_3', 'burst_6', 'noise_0.15', 'noise_0.3']


@torch.no_grad()
def corrupt(images_uint8, condition, generator):
    """(T,3,H,W) uint8 on device -> float [0,1] images and a (T,) bool 'corrupted' mask."""
    x = images_uint8.float() / 255.0
    t, c, h, w = x.shape
    dev = x.device
    hit = torch.zeros(t, dtype=torch.bool, device=dev)
    if condition == 'clean':
        return x, hit
    kind, level = condition.split('_')
    if kind == 'dropout':
        hit = torch.rand(t, device=dev, generator=generator) < float(level)
        x[hit] = 0.5
    elif kind == 'noise':
        x = (x + float(level) * torch.randn(x.shape, device=dev, generator=generator)).clamp(0, 1)
        hit[:] = True
    elif kind == 'burst':
        k = int(level)
        n_bursts = max(1, round(0.3 * t / k))
        for _ in range(n_bursts):
            s = int(torch.randint(0, max(1, t - k + 1), (1,), device=dev, generator=generator))
            bh = int(h * (0.4 + 0.2 * float(torch.rand(1, device=dev, generator=generator))))
            bw = int(w * (0.4 + 0.2 * float(torch.rand(1, device=dev, generator=generator))))
            y0 = int(torch.randint(0, h - bh + 1, (1,), device=dev, generator=generator))
            x0 = int(torch.randint(0, w - bw + 1, (1,), device=dev, generator=generator))
            fill = torch.rand(3, 1, 1, device=dev, generator=generator)
            x[s:s + k, :, y0:y0 + bh, x0:x0 + bw] = fill
            hit[s:s + k] = True
    return x, hit


@torch.no_grad()
def predict(model, x, chunk=32):
    preds = {k: [] for k in ('seg', 'bbox', 'bbox_prob', 'line', 'line_vis_prob', 'pose')}
    state = None
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            if isinstance(model, RowCNNLSTM):
                outs = []
                for i in range(len(xc)):
                    o, state = model.step(xc[i:i + 1], state)
                    outs.append(o)
                out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
            else:
                out = model(xc)
        preds['seg'].append(out['seg'].argmax(1).byte().cpu())
        preds['bbox'].append(out['bbox'].float().cpu())
        preds['bbox_prob'].append(torch.sigmoid(out['bbox_logit'].float()).cpu())
        preds['line'].append(out['line'].float().cpu())
        preds['line_vis_prob'].append(torch.sigmoid(out['line_vis_logit'].float()).cpu())
        preds['pose'].append(out['pose'].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in preds.items()}


def subset(seq, pred, mask):
    """Restrict a sequence + prediction to frames in mask (temporal metrics then
    only use consecutive pairs that are both in the subset -- approximated by
    concatenation, so jitter/flicker are not reported for subsets)."""
    idx = np.flatnonzero(mask)
    s = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in seq.items()}
    p = {k: v[idx] for k, v in pred.items()}
    return s, p


def bootstrap_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATASET))
    ap.add_argument('--runs', default=str(MODELS_DIR / 'row_net'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--models', nargs=2, default=['cnn', 'cnn_lstm'], help='baseline, candidate')
    ap.add_argument('--conditions', nargs='+', default=CONDITIONS)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Sequences(args.data, args.split)
    models = {k: load(k, args.runs, device) for k in args.models}
    base, cand = args.models
    report = {}

    for condition in args.conditions:
        per_seq = {k: [] for k in args.models}           # list of per-sequence summaries
        pooled = {k: Metrics() for k in args.models}      # all frames
        pooled_hit = {k: Metrics() for k in args.models}  # corrupted frames only
        for seq in data.seqs:
            images = torch.from_numpy(seq['images']).to(device).permute(0, 3, 1, 2)
            # same corruption for both models
            x, hit = corrupt(images, condition, seq_generator(seq['name'] + condition, device))
            hit = hit.cpu().numpy()
            for k, model in models.items():
                p = predict(model, x)
                m = Metrics(); m.add_sequence(seq, p)
                per_seq[k].append(m.summary())
                pooled[k].add_sequence(seq, p)
                if condition != 'clean' and hit.any():
                    pooled_hit[k].add_sequence(*subset(seq, p, hit))

        cond_report = {'all_frames': {}, 'corrupted_frames': {}}
        for metric in KEY_METRICS:
            b = np.array([s[metric] for s in per_seq[base]])
            c = np.array([s[metric] for s in per_seq[cand]])
            ok = ~(np.isnan(b) | np.isnan(c))
            d = c[ok] - b[ok]
            lower = any(t in metric for t in LOWER_IS_BETTER)
            wins = int(((d < 0) if lower else (d > 0)).sum())
            lo, hi = bootstrap_ci(d)
            cond_report['all_frames'][metric] = {
                base: pooled[base].summary()[metric], cand: pooled[cand].summary()[metric],
                'mean_seq_diff': float(d.mean()), 'ci95': [lo, hi],
                'significant': bool(lo > 0 or hi < 0),
                'candidate_better': bool((d.mean() < 0) if lower else (d.mean() > 0)),
                'candidate_wins': wins, 'n_seq': int(ok.sum()),
            }
            if condition != 'clean' and 'jitter' not in metric and 'flicker' not in metric:
                cond_report['corrupted_frames'][metric] = {
                    base: pooled_hit[base].summary()[metric], cand: pooled_hit[cand].summary()[metric]}
        report[condition] = cond_report
        print(f'[analyze] {condition}: ' + ', '.join(
            f"{m} {v[base]:.3f}->{v[cand]:.3f}{'*' if v['significant'] else ''}"
            for m, v in cond_report['all_frames'].items()), flush=True)

    tag = f'{args.split}' if args.models == ['cnn', 'cnn_lstm'] else f'{args.split}_{base}_vs_{cand}'
    json.dump(report, open(os.path.join(args.runs, f'analysis_{tag}.json'), 'w'), indent=2)

    lines = [f'# {cand} vs {base}: paired analysis ({args.split}, {len(data)} sequences)', '',
             'Mean per-sequence difference (candidate - baseline) with 95% bootstrap CI over sequences.',
             '**bold** = CI excludes 0. Wins = sequences where the candidate is better.', '']
    for condition, r in report.items():
        lines += [f'## {condition}', '', f'| metric | {base} | {cand} | diff [95% CI] | wins |', '|---|---|---|---|---|']
        for metric, v in r['all_frames'].items():
            diff = f"{v['mean_seq_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
            diff = f'**{diff}** ({"better" if v["candidate_better"] else "worse"})' if v['significant'] else diff
            lines.append(f"| {metric} | {v[base]:.4f} | {v[cand]:.4f} | {diff} | {v['candidate_wins']}/{v['n_seq']} |")
        if r['corrupted_frames']:
            lines += ['', 'Corrupted frames only:', '', f'| metric | {base} | {cand} |', '|---|---|---|']
            for metric, v in r['corrupted_frames'].items():
                lines.append(f'| {metric} | {v[base]:.4f} | {v[cand]:.4f} |')
        lines.append('')
    open(os.path.join(args.runs, f'analysis_{tag}.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
