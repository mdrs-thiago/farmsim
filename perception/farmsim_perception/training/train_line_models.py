#!/usr/bin/env python3
"""Train and compare frame-wise CNN and CNN-LSTM line estimators."""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from farmsim_perception.models.line_models import make_model
from farmsim_perception.line_data import LineSequenceDataset, find_sessions, official_splits, split_sessions
from farmsim_utils.paths import DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def masked_loss(prediction, target, visible):
    coordinate_mask = visible.repeat_interleave(2, dim=-1)
    if not coordinate_mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[coordinate_mask], target[coordinate_mask])


def angle(lines, width, height):
    du = (lines[..., 2] - lines[..., 0]) * width / 2.0
    up = (lines[..., 1] - lines[..., 3]) * height / 2.0
    return torch.rad2deg(torch.atan2(du, up))


@torch.no_grad()
def evaluate(model, loader, device, width, height):
    model.eval()
    near_errors, far_errors, angle_errors, delta_angle_errors = [], [], [], []
    losses = []
    seen_frames, seen_pairs = set(), set()
    for batch in loader:
        images = batch['images'].to(device)
        target = batch['line'].to(device)
        visible = batch['visible'].to(device)
        output = model(images)
        prediction = output['line'] if isinstance(output, dict) else output
        # The final clip of a session can overlap the preceding clip. Count
        # every held-out frame/pair once so metrics are not biased toward the
        # last few frames of each recording.
        fresh = torch.zeros_like(visible[..., 0])
        fresh_pairs = torch.zeros_like(visible[:, 1:, 0])
        for b, session in enumerate(batch['session']):
            indices = batch['frame_index'][b].tolist()
            for t, index in enumerate(indices):
                key = (session, int(index))
                if key not in seen_frames:
                    fresh[b, t] = True; seen_frames.add(key)
            for t, (left, right) in enumerate(zip(indices, indices[1:])):
                key = (session, int(left), int(right))
                if key not in seen_pairs:
                    fresh_pairs[b, t] = True; seen_pairs.add(key)
        losses.append(masked_loss(prediction, target, visible & fresh.unsqueeze(-1)).item())
        scale = torch.tensor([width / 2, height / 2, width / 2, height / 2], device=device)
        error = (prediction - target).abs() * scale
        near = torch.linalg.vector_norm(error[..., :2], dim=-1)
        far = torch.linalg.vector_norm(error[..., 2:], dim=-1)
        near_errors.extend(near[visible[..., 0] & fresh].cpu().tolist())
        far_errors.extend(far[visible[..., 1] & fresh].cpu().tolist())
        both = visible.all(dim=-1) & fresh
        pred_angle, true_angle = angle(prediction, width, height), angle(target, width, height)
        ae = torch.remainder(pred_angle - true_angle + 180.0, 360.0).sub(180.0).abs()
        angle_errors.extend(ae[both].cpu().tolist())
        consecutive = visible[:, 1:].all(dim=-1) & visible[:, :-1].all(dim=-1) & fresh_pairs
        pred_delta = pred_angle[:, 1:] - pred_angle[:, :-1]
        true_delta = true_angle[:, 1:] - true_angle[:, :-1]
        de = torch.remainder(pred_delta - true_delta + 180.0, 360.0).sub(180.0).abs()
        delta_angle_errors.extend(de[consecutive].cpu().tolist())

    def mean(values): return float(np.mean(values)) if values else float('nan')
    def p90(values): return float(np.percentile(values, 90)) if values else float('nan')
    return {
        'loss': mean(losses),
        'near_mae_px': mean(near_errors), 'near_p90_px': p90(near_errors),
        'far_mae_px': mean(far_errors), 'far_p90_px': p90(far_errors),
        'angle_mae_deg': mean(angle_errors),
        'temporal_delta_angle_mae_deg': mean(delta_angle_errors),
        'near_n': len(near_errors), 'far_n': len(far_errors),
        'both_n': len(angle_errors), 'consecutive_both_n': len(delta_angle_errors),
    }


class ConstantLineModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__(); self.register_buffer('value', value)

    def forward(self, images):
        return self.value.view(1, 1, 4).expand(images.shape[0], images.shape[1], 4)


def training_mean_line(dataset):
    points = [[], []]
    for sequence in dataset.sequences:
        for frame in sequence:
            for endpoint in (0, 1):
                if frame['visible'][endpoint]:
                    points[endpoint].append(frame['line'][2 * endpoint:2 * endpoint + 2])
    means = [np.mean(values, axis=0) if values else np.zeros(2) for values in points]
    return torch.tensor(np.concatenate(means), dtype=torch.float32)


def train_one(kind, datasets, args, device):
    seed_everything(args.seed)
    model = make_model(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    history, best_state, best_loss, stale = [], None, float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        started = time.time(); model.train(); total = batches = 0
        for batch in train_loader:
            prediction = model(batch['images'].to(device))
            loss = masked_loss(prediction, batch['line'].to(device), batch['visible'].to(device))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); total += loss.item(); batches += 1
        val = evaluate(model, val_loader, device, *args.metric_image_size)
        row = {'epoch': epoch, 'train_loss': total / max(batches, 1), 'val': val,
               'elapsed_s': time.time() - started}
        history.append(row)
        print(f'{kind:8s} epoch {epoch:02d}: train={row["train_loss"]:.5f} '
              f'val={val["loss"]:.5f} near={val["near_mae_px"]:.1f}px '
              f'far={val["far_mae_px"]:.1f}px angle={val["angle_mae_deg"]:.2f}deg '
              f'dt-angle={val["temporal_delta_angle_mae_deg"]:.2f}deg '
              f'({row["elapsed_s"]:.1f}s)', flush=True)
        if val['loss'] < best_loss - 1e-5:
            best_loss = val['loss']; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f'{kind}: early stop after {epoch} epochs', flush=True)
                break
    model.load_state_dict(best_state)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test = evaluate(model, test_loader, device, *args.metric_image_size)
    out_dir = os.path.join(args.out_dir, kind)
    os.makedirs(out_dir, exist_ok=True)
    split_names = {key: [os.path.basename(s) for s in value]
                   for key, value in args.session_splits.items()}
    torch.save({'kind': kind, 'state_dict': best_state, 'image_size': args.image_size,
                'sequence_length': args.sequence_length, 'splits': split_names,
                'test_metrics': test}, os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'kind': kind, 'parameters': sum(p.numel() for p in model.parameters()),
                   'history': history, 'test': test, 'splits': split_names}, f, indent=2)
        f.write('\n')
    print(f'{kind} TEST: {json.dumps(test)}', flush=True)
    return {'kind': kind, 'parameters': sum(p.numel() for p in model.parameters()), 'test': test}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', default=str(DEFAULT_DATASET))
    ap.add_argument('--models', nargs='+', choices=('cnn', 'cnn_lstm'), default=('cnn', 'cnn_lstm'))
    ap.add_argument('--sequence-length', type=int, default=8)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--image-size', type=int, nargs=2, default=(96, 72), metavar=('WIDTH', 'HEIGHT'))
    ap.add_argument('--metric-image-size', type=int, nargs=2, default=(640, 480),
                    metavar=('WIDTH', 'HEIGHT'), help='pixel units used for reported errors')
    ap.add_argument('--no-preload', action='store_true',
                    help='decode images per clip instead of caching resized RGB frames in RAM')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader workers (0 is portable in restricted/container environments)')
    ap.add_argument('--seed', type=int, default=20260913)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out-dir', default=str(MODELS_DIR / 'line_benchmark'))
    args = ap.parse_args()
    args.image_size = tuple(args.image_size)
    args.metric_image_size = tuple(args.metric_image_size)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    sessions = find_sessions(args.dataset_root)
    train, val, test = (official_splits(args.dataset_root, sessions)
                        or split_sessions(sessions, seed=args.seed))
    args.session_splits = {'train': train, 'val': val, 'test': test}
    datasets = {name: LineSequenceDataset(split,
                                          args.sequence_length, args.stride, args.image_size,
                                          preload=not args.no_preload)
                for name, split in args.session_splits.items()}
    print(f'device={device} sessions train/val/test={len(train)}/{len(val)}/{len(test)}')
    for name, dataset in datasets.items():
        print(f'  {name}: clips={len(dataset)} visibility={dataset.visibility_counts()} '
              f'skipped_sessions={len(dataset.skipped_sessions)}')
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise SystemExit('One split has no usable sequence clips')
    os.makedirs(args.out_dir, exist_ok=True)
    results = [train_one(kind, datasets, args, device) for kind in args.models]
    baseline_model = ConstantLineModel(training_mean_line(datasets['train'])).to(device)
    baseline_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    baseline = evaluate(baseline_model, baseline_loader, device, *args.metric_image_size)
    with open(os.path.join(args.out_dir, 'comparison.json'), 'w') as f:
        json.dump({'config': vars(args) | {'device': str(device)},
                   'constant_training_mean_baseline': baseline, 'models': results}, f, indent=2)
        f.write('\n')
    print('COMPARISON')
    print(f"  baseline near={baseline['near_mae_px']:.2f}px far={baseline['far_mae_px']:.2f}px "
          f"angle={baseline['angle_mae_deg']:.2f}deg "
          f"temporal-delta={baseline['temporal_delta_angle_mae_deg']:.2f}deg")
    for result in results:
        t = result['test']
        print(f"  {result['kind']:8s} near={t['near_mae_px']:.2f}px far={t['far_mae_px']:.2f}px "
              f"angle={t['angle_mae_deg']:.2f}deg temporal-delta={t['temporal_delta_angle_mae_deg']:.2f}deg")


if __name__ == '__main__':
    main()
