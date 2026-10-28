#!/usr/bin/env python3
"""
evaluate.py

Compares trained checkpoints on the test split, on clean frames and on corrupted
frames (the training augmentation replayed with a fixed seed per sequence:
sensor noise, occluders, exposure spikes). Writes:

  runs/comparison_<split>.json     all metrics
  runs/comparison_<split>.md       side-by-side table (CNN-LSTM both stateful and 8-frame window)
  runs/qualitative_<sequence>.jpg  consecutive corrupted frames: input | GT | CNN | CNN-LSTM

Usage:
  python -m farmsim_perception.training.evaluate
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from farmsim_perception.data import Sequences, augment_batch
from farmsim_perception.engine import Metrics, predict_sequence, seq_generator
from farmsim_perception.models.row_net import make_model

from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401
COLORS_RGB = np.array([(0, 0, 0), (40, 200, 40), (40, 90, 255), (255, 200, 0)], np.uint8)

LOWER_IS_BETTER = ('err', 'MAE', 'jitter', 'flicker')


def load(run_name, run_root, device):
    ckpt = torch.load(os.path.join(run_root, run_name, 'last.pt'), map_location=device)
    model = make_model(ckpt['model'], pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval()


def qualitative(models, seq, device, path, start=None, n=6):
    images = torch.from_numpy(seq['images']).to(device).permute(0, 3, 1, 2)
    x = augment_batch(images.unsqueeze(0), generator=seq_generator(seq['name'], device))[0]
    preds = {k: predict_sequence(m, seq, device, corrupt=True) for k, m in models.items()}
    start = len(images) // 2 if start is None else start
    rows = []
    for t in range(start, min(start + n, len(images))):
        inp = (x[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        tiles = [inp, COLORS_RGB[seq['masks'][t]]]
        for k in models:
            tile = cv2.addWeighted(inp, 0.35, COLORS_RGB[preds[k]['seg'][t]], 0.65, 0)
            u0, v0, u1, v1 = preds[k]['line'][t]
            w, h = tile.shape[1], tile.shape[0]
            cv2.arrowedLine(tile, (int((u0 + 1) * w / 2), int((v0 + 1) * h / 2)),
                            (int((u1 + 1) * w / 2), int((v1 + 1) * h / 2)), (255, 0, 255), 2)
            cv2.putText(tile, k, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def selected_frames(seq, pred, selected):
    """Select frames for spatial/error metrics; adjacent-frame metrics are excluded."""
    sliced_seq = {k: v[selected] if isinstance(v, np.ndarray) and len(v) == len(selected) else v
                  for k, v in seq.items()}
    sliced_pred = {k: v[selected] for k, v in pred.items()}
    return sliced_seq, sliced_pred


def paired_interval(values, seed=0, n_bootstrap=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n_sequences': 0, 'mean_sequence_change': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_bootstrap, len(values)))].mean(axis=1)
    return {'n_sequences': len(values), 'mean_sequence_change': float(values.mean()),
            'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATASET))
    ap.add_argument('--runs', default=str(MODELS_DIR / 'row_net'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--models', nargs='+', default=['cnn', 'cnn_lstm'])
    ap.add_argument('--variant', default=None, help='anomaly variant root from make_anomaly_variant.py')
    ap.add_argument('--limit-sequences', type=int, default=None, help='debug: evaluate first N sequences')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Sequences(args.data, args.split, limit=args.limit_sequences,
                     variant_root=args.variant, include_source_images=bool(args.variant))
    models = {k: load(k, args.runs, device) for k in args.models}
    print(f'[eval] {args.split}: {len(data)} sequences, {data.n_frames} frames', flush=True)

    # CNN-LSTM is scored twice: stateful over the whole sequence (how it runs on the
    # robot) and with an 8-frame sliding window (the clip length it was trained on)
    variants = []
    for kind, model in models.items():
        variants.append((kind, model, None))
        if model.kind == 'cnn_lstm':
            variants.append((f'{kind}_window8', model, 8))

    results = {}
    subset_metrics = ('seg_mIoU', 'det_F1@0.5', 'line_near_err_px', 'line_far_err_px',
                      'lateral_MAE_cm', 'heading_MAE_deg')
    for condition in ('clean', 'corrupted'):
        for name, model, window in variants:
            m = Metrics()
            affected, unaffected, affected_source = Metrics(), Metrics(), Metrics()
            affected_n = unaffected_n = 0
            sequence_effects = {k: [] for k in subset_metrics}
            for seq in data.seqs:
                pred = predict_sequence(model, seq, device, corrupt=condition == 'corrupted', window=window)
                m.add_sequence(seq, pred)
                if args.variant:
                    exposed = np.any(seq['anomaly_masks'] != 0, axis=(1, 2))
                    for selected, target in ((exposed, affected), (~exposed, unaffected)):
                        if selected.any():
                            target.add_sequence(*selected_frames(seq, pred, selected))
                    if exposed.any():
                        source_seq = {**seq, 'images': seq['source_images']}
                        source_pred = predict_sequence(model, source_seq, device,
                                                       corrupt=condition == 'corrupted', window=window)
                        variant_slice = selected_frames(seq, pred, exposed)
                        source_slice = selected_frames(source_seq, source_pred, exposed)
                        affected_source.add_sequence(*source_slice)
                        per_variant, per_source = Metrics(), Metrics()
                        per_variant.add_sequence(*variant_slice)
                        per_source.add_sequence(*source_slice)
                        variant_summary, source_summary = per_variant.summary(), per_source.summary()
                        for metric in subset_metrics:
                            if np.isfinite(variant_summary[metric]) and np.isfinite(source_summary[metric]):
                                sequence_effects[metric].append(variant_summary[metric] - source_summary[metric])
                    affected_n += int(exposed.sum())
                    unaffected_n += int((~exposed).sum())
            results.setdefault(condition, {})[name] = m.summary()
            if args.variant:
                affected_result = ({k: affected.summary()[k] for k in subset_metrics} if affected_n else {})
                source_result = ({k: affected_source.summary()[k] for k in subset_metrics}
                                 if affected_n else {})
                results.setdefault('anomaly_presence', {}).setdefault(condition, {})[name] = {
                    'affected': {'frames': affected_n, **affected_result},
                    'source_same_frames': {'frames': affected_n, **source_result},
                    'effect_on_affected_frames': {k: affected_result[k] - source_result[k]
                                                  for k in subset_metrics} if affected_n else {},
                    'paired_sequence_effect': {k: paired_interval(v) for k, v in sequence_effects.items()},
                    'unaffected': {'frames': unaffected_n, **({k: unaffected.summary()[k] for k in subset_metrics}
                                                           if unaffected_n else {})},
                }
            print(f'[eval] {condition} {name}: ' + json.dumps({k: round(v, 4) for k, v in results[condition][name].items()}), flush=True)

    tag = args.split + (f'_{os.path.basename(os.path.normpath(args.variant))}' if args.variant else '')
    if args.limit_sequences is not None:
        tag += f'_first{args.limit_sequences}'
    json.dump(results, open(os.path.join(args.runs, f'comparison_{tag}.json'), 'w'), indent=2)

    lines = [f'# Row-perception comparison ({tag}: {len(data)} sequences, {data.n_frames} frames)', '']
    for condition in ('clean', 'corrupted'):
        res = results[condition]
        kinds = list(res)
        others = kinds[1:]
        columns = ['metric', *kinds, *(f'{k} vs {kinds[0]}' for k in others)]
        lines += [f'## {condition}', '', '| ' + ' | '.join(columns) + ' |',
                  '|---|' + '---|' * (len(columns) - 1)]
        for metric in res[kinds[0]]:
            base = res[kinds[0]][metric]
            changes = []
            for k in others:
                if not np.isfinite(base) or not np.isfinite(res[k][metric]) or not base:
                    changes.append('n/a'); continue
                rel = 100 * (res[k][metric] - base) / abs(base)
                better = (rel < 0) if any(s in metric for s in LOWER_IS_BETTER) else (rel > 0)
                changes.append(f'{rel:+.1f}% {"better" if better else "worse"}')
            values = [metric, *(f'{res[k][metric]:.4f}' for k in kinds), *changes]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines.append('')
    if args.variant:
        lines += ['## Anomaly exposure', '',
                  'Paired error on the same frames, using source images and anomaly images.',
                  'Temporal jitter and flicker are omitted because selected frames need not be consecutive.', '']
        for condition in ('clean', 'corrupted'):
            lines += [f'### {condition}', '',
                      '| model | affected frames | unaffected frames | source far error (px) | anomaly far error (px) | pooled change (px) | mean sequence change [95% CI] (px) |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for name, groups in results['anomaly_presence'][condition].items():
                a, u = groups['affected'], groups['unaffected']
                far = f'{a["line_far_err_px"]:.2f}' if a['frames'] else 'n/a'
                source = f'{groups["source_same_frames"]["line_far_err_px"]:.2f}' if a['frames'] else 'n/a'
                delta = f'{groups["effect_on_affected_frames"]["line_far_err_px"]:+.2f}' if a['frames'] else 'n/a'
                paired = groups['paired_sequence_effect']['line_far_err_px']
                interval = (f'{paired["mean_sequence_change"]:+.2f} '
                            f'[{paired["ci95"][0]:+.2f}, {paired["ci95"][1]:+.2f}] '
                            f'(n={paired["n_sequences"]})') if paired['n_sequences'] else 'n/a'
                lines.append(f'| {name} | {a["frames"]} | {u["frames"]} | {source} | {far} | {delta} | {interval} |')
            lines.append('')
    open(os.path.join(args.runs, f'comparison_{tag}.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))

    for seq in data.seqs[:3]:
        qualitative(models, seq, device, os.path.join(args.runs, f'qualitative_{tag}_{seq["name"]}.jpg'))


if __name__ == '__main__':
    main()
