#!/usr/bin/env python3
"""
train.py

Trains RowCNN or RowCNNLSTM on datasets/thorvald_row_perception_v1 with an equal
budget: both models process the same number of training frames per epoch
(CNN: shuffled single frames; CNN-LSTM: shuffled clips of --clip-len consecutive
frames), with the same augmentation, optimiser and schedule. The final-epoch
weights are the ones evaluated, so there is no val-based checkpoint picking that
could favour either model.

Usage:
  python -m farmsim_perception.training.train --model cnn --encoder-init scratch
  python -m farmsim_perception.training.train --model cnn_lstm --encoder-init imagenet
"""

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from farmsim_perception.data import ClipDataset, FrameDataset, Sequences, augment_batch
from farmsim_perception.engine import evaluate, multitask_loss
from farmsim_perception.models.row_net import make_model

from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['cnn', 'cnn_lstm'], required=True)
    ap.add_argument('--encoder-init', choices=['scratch', 'imagenet'], default='imagenet',
                    help='ResNet-18 encoder initialization (default: imagenet for existing runs)')
    ap.add_argument('--data', default=str(DEFAULT_DATASET))
    ap.add_argument('--train-variant', default=None, help='anomaly variant root for training images')
    ap.add_argument('--val-variant', default=None, help='anomaly variant root for validation images')
    ap.add_argument('--out', default=str(MODELS_DIR / 'row_net'))
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--frames-per-batch', type=int, default=32)
    ap.add_argument('--clip-len', type=int, default=8)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--encoder-lr-scale', type=float, default=0.3)
    ap.add_argument('--limit-sequences', type=int, default=None, help='debug: use only N sequences per split')
    ap.add_argument('--max-steps', type=int, default=None, help='debug: stop each epoch after N steps')
    ap.add_argument('--workers', type=int, default=0, help='>0 copies the in-RAM dataset into every worker (forkserver)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init-from', default=None,
                    help='checkpoint to start from; matching weights are loaded, new layers (e.g. ConvLSTM) keep their init')
    ap.add_argument('--run-name', default=None,
                    help='output subdirectory (default: model/init name, with _variant for variant training)')
    ap.add_argument('--warmup-steps', type=int, default=500)
    ap.add_argument('--freeze-bn', action='store_true',
                    help='keep BatchNorm running stats of the loaded layers fixed (clip batches are highly correlated)')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    default_run_name = args.model if args.encoder_init == 'imagenet' else f'{args.model}_scratch'
    if args.train_variant or args.val_variant:
        default_run_name += '_variant'
    run_name = args.run_name or default_run_name
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    train = Sequences(args.data, 'train', limit=args.limit_sequences, variant_root=args.train_variant)
    val = Sequences(args.data, 'val', limit=args.limit_sequences, variant_root=args.val_variant)
    print(f'[data] train {len(train)} seq / {train.n_frames} frames, val {len(val)} seq / {val.n_frames} frames '
          f'({time.time() - t0:.0f}s)', flush=True)

    if args.model == 'cnn':
        ds, batch_size, n_samples = FrameDataset(train), args.frames_per_batch, train.n_frames
    else:
        ds = ClipDataset(train, clip_len=args.clip_len, stride=max(1, args.clip_len // 4))
        batch_size = max(1, args.frames_per_batch // args.clip_len)
        n_samples = train.n_frames // args.clip_len
    sampler = RandomSampler(ds, replacement=n_samples > len(ds), num_samples=n_samples,
                            generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=True,
                        persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = min(len(loader), args.max_steps or len(loader))
    total_steps = steps_per_epoch * args.epochs

    model = make_model(args.model, pretrained=args.encoder_init == 'imagenet').to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)['state_dict']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[init] {args.init_from}: {len(state) - len(unexpected)} tensors loaded, '
              f'{len(missing)} new ({sorted({k.split(".")[0] for k in missing})})', flush=True)
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': args.lr * args.encoder_lr_scale},
                             {'params': rest, 'lr': args.lr}], weight_decay=1e-4)
    base_lrs = [g['lr'] for g in opt.param_groups]
    warmup = min(args.warmup_steps, total_steps // 10)

    # Model constructors consume different amounts of randomness. Reset before
    # augmentation so paired runs with the same seed see the same random stream.
    torch.manual_seed(args.seed)

    def set_lr(step):
        f = step / max(warmup, 1) if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * max(f, 0.01)

    # background (sky + outside the field) is ~43% of pixels and easy; the three field classes are 16-22% each
    class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0], device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] {args.model} / {args.encoder_init}: {n_params / 1e6:.2f}M params, batch {batch_size}x'
          f'{1 if args.model == "cnn" else args.clip_len}, {steps_per_epoch} steps/epoch', flush=True)

    history, step = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.BatchNorm2d) and not name.startswith(('t16', 't32')):
                    mod.eval()
        t_epoch, running, n = time.time(), {}, 0
        for i, batch in enumerate(loader):
            if i >= steps_per_epoch:
                break
            set_lr(step)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            x = augment_batch(batch['image'])
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                out = model(x)
            if x.dim() == 5:  # flatten (B,T,...) -> (B*T,...) for the per-frame losses
                out = {k: v.flatten(0, 1) for k, v in out.items()}
                batch = {k: v.flatten(0, 1) for k, v in batch.items()}
            loss, parts = multitask_loss(out, batch, class_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step += 1
            n += 1
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v
            if i % 100 == 0:
                print(f'  ep {epoch} step {i}/{steps_per_epoch} loss {loss.item():.3f} '
                      + ' '.join(f'{k}={v:.3f}' for k, v in parts.items()), flush=True)

        train_time = time.time() - t_epoch
        metrics = evaluate(model, val, device)
        record = {'epoch': epoch, 'train_loss': {k: v / max(n, 1) for k, v in running.items()},
                  'train_time_s': train_time, 'val': metrics}
        history.append(record)
        print(f'[epoch {epoch}] {train_time:.0f}s  val mIoU {metrics["seg_mIoU"]:.3f} flicker {metrics["seg_flicker_pct"]:.2f}% '
              f'detF1 {metrics["det_F1@0.5"]:.3f} line far {metrics["line_far_err_px"]:.1f}px '
              f'lat {metrics["lateral_MAE_cm"]:.1f}cm head {metrics["heading_MAE_deg"]:.2f}deg', flush=True)
        torch.save({'model': args.model, 'state_dict': model.state_dict(), 'epoch': epoch, 'args': vars(args)},
                   os.path.join(run_dir, 'last.pt'))
        json.dump(history, open(os.path.join(run_dir, 'history.json'), 'w'), indent=2)

    print(f'[done] {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
