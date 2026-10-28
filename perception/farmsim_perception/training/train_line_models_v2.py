#!/usr/bin/env python3
"""Train spatial heatmap CNN and CNN-LSTM line estimators."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from farmsim_perception.models.line_models_v2 import make_model
from farmsim_perception.line_data import LineSequenceDataset, find_sessions, official_splits, split_sessions
from farmsim_utils.paths import DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR
from farmsim_perception.training.train_line_models import evaluate


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def endpoint_balanced_loss(prediction, target, visible):
    losses = []
    for endpoint in (0, 1):
        mask = visible[..., endpoint]
        if mask.any():
            sl = slice(2 * endpoint, 2 * endpoint + 2)
            losses.append(F.smooth_l1_loss(prediction[..., sl][mask], target[..., sl][mask]))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def heatmap_loss(logits, target, visible):
    _, _, _, height, width = logits.shape
    losses = []
    for endpoint in (0, 1):
        mask = visible[..., endpoint]
        if not mask.any():
            continue
        x = ((target[..., 2 * endpoint].clamp(-1, 1) + 1) * 0.5 * (width - 1)).round().long()
        y = ((target[..., 2 * endpoint + 1].clamp(-1, 1) + 1) * 0.5 * (height - 1)).round().long()
        index = y * width + x
        endpoint_logits = logits[:, :, endpoint].flatten(2)
        losses.append(F.cross_entropy(endpoint_logits[mask], index[mask]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def temporal_delta_loss(prediction, target, visible):
    losses = []
    for endpoint in (0, 1):
        mask = visible[:, 1:, endpoint] & visible[:, :-1, endpoint]
        if mask.any():
            sl = slice(2 * endpoint, 2 * endpoint + 2)
            pred_delta = prediction[:, 1:, sl] - prediction[:, :-1, sl]
            true_delta = target[:, 1:, sl] - target[:, :-1, sl]
            losses.append(F.smooth_l1_loss(pred_delta[mask], true_delta[mask]))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def total_loss(output, target, visible):
    coordinate = endpoint_balanced_loss(output['line'], target, visible)
    heatmap = heatmap_loss(output['heatmap_logits'], target, visible)
    temporal = temporal_delta_loss(output['line'], target, visible)
    return coordinate + 0.02 * heatmap + 0.20 * temporal, {
        'coordinate': coordinate.item(), 'heatmap': heatmap.item(), 'temporal': temporal.item()}


def train_one(kind, datasets, split_paths, args, device):
    seed_everything(args.seed)
    model = make_model(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    best_score, best_state, stale, history = float('inf'), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); started = time.time(); sums = np.zeros(4); batches = 0
        for batch in train_loader:
            images = batch['images'].to(device); target = batch['line'].to(device)
            visible = batch['visible'].to(device); output = model(images)
            loss, parts = total_loss(output, target, visible)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            sums += [loss.item(), parts['coordinate'], parts['heatmap'], parts['temporal']]
            batches += 1
        metrics = evaluate(model, val_loader, device, *args.metric_image_size)
        score = metrics['near_mae_px'] + metrics['far_mae_px'] + 2.0 * metrics['angle_mae_deg']
        scheduler.step(score)
        row = {'epoch': epoch, 'train_total': float(sums[0] / batches),
               'train_coordinate': float(sums[1] / batches),
               'train_heatmap': float(sums[2] / batches),
               'train_temporal': float(sums[3] / batches), 'val_score': score,
               'val': metrics, 'lr': optimizer.param_groups[0]['lr'],
               'elapsed_s': time.time() - started}
        history.append(row)
        print(f'{kind:16s} epoch {epoch:02d}: coord={row["train_coordinate"]:.4f} '
              f'heat={row["train_heatmap"]:.3f} val-score={score:.1f} '
              f'near={metrics["near_mae_px"]:.1f}px far={metrics["far_mae_px"]:.1f}px '
              f'angle={metrics["angle_mae_deg"]:.1f}deg '
              f'delta={metrics["temporal_delta_angle_mae_deg"]:.1f}deg '
              f'({row["elapsed_s"]:.1f}s)', flush=True)
        if score < best_score - 0.1:
            best_score = score; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f'{kind}: early stop at epoch {epoch}', flush=True); break

    model.load_state_dict(best_state)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test = evaluate(model, test_loader, device, *args.metric_image_size)
    split_names = {key: [os.path.basename(path) for path in value]
                   for key, value in split_paths.items()}
    out = os.path.join(args.out_dir, kind); os.makedirs(out, exist_ok=True)
    torch.save({'kind': kind, 'state_dict': best_state, 'image_size': args.image_size,
                'sequence_length': args.sequence_length, 'splits': split_names,
                'test_metrics': test}, os.path.join(out, 'model.pt'))
    result = {'kind': kind, 'parameters': sum(p.numel() for p in model.parameters()),
              'best_val_score': best_score, 'history': history, 'test': test,
              'splits': split_names}
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(f'{kind} TEST {json.dumps(test)}', flush=True)
    return {k: result[k] for k in ('kind', 'parameters', 'best_val_score', 'test')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', default=str(DEFAULT_DATASET))
    ap.add_argument('--models', nargs='+', choices=('spatial_cnn', 'spatial_cnn_lstm'),
                    default=('spatial_cnn', 'spatial_cnn_lstm'))
    ap.add_argument('--sequence-length', type=int, default=8); ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--image-size', type=int, nargs=2, default=(96, 72))
    ap.add_argument('--metric-image-size', type=int, nargs=2, default=(640, 480))
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=32); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--cache-dir', default=str(OUTPUTS_DIR / 'line_frame_cache'))
    ap.add_argument('--out-dir', default=str(MODELS_DIR / 'line_benchmark_v2'))
    args = ap.parse_args(); args.image_size = tuple(args.image_size)
    args.metric_image_size = tuple(args.metric_image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sessions = find_sessions(args.dataset_root)
    train, val, test = (official_splits(args.dataset_root, sessions)
                        or split_sessions(sessions, seed=args.seed))
    split_paths = {'train': train, 'val': val, 'test': test}
    datasets = {name: LineSequenceDataset(paths, args.sequence_length,
                                          args.stride, args.image_size, preload=True,
                                          cache_dir=args.cache_dir, augment=name == 'train')
                for name, paths in split_paths.items()}
    print(f'device={device} sessions={len(train)}/{len(val)}/{len(test)}')
    for name, dataset in datasets.items(): print(name, len(dataset), dataset.visibility_counts())
    results = [train_one(kind, datasets, split_paths, args, device) for kind in args.models]
    old_path = os.path.join(os.path.dirname(args.out_dir), 'line_benchmark', 'comparison.json')
    old = json.load(open(old_path)) if os.path.isfile(old_path) else None
    comparison = {'config': vars(args) | {'device': str(device)}, 'v2_models': results,
                  'v1_comparison': old}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2); f.write('\n')
    print('V2 COMPARISON')
    for result in results:
        t = result['test']; print(result['kind'], t)


if __name__ == '__main__': main()
