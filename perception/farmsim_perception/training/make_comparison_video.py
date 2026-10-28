#!/usr/bin/env python3
"""
make_comparison_video.py

Side-by-side video of the CNN and the CNN-LSTM running over a test sequence in
temporal order, under a chosen corruption (see analyze.py CONDITIONS). Each
frame shows, for both models, the segmentation overlay, the predicted row-bed
box and the predicted row line, plus the running segmentation IoU and the
lateral-offset error, so the difference is visible rather than tabulated.

The CNN-LSTM keeps its recurrent state across the whole clip, as it would on the
robot; the CNN sees each frame independently. Both get exactly the same input.

Usage:
  python -m farmsim_perception.training.make_comparison_video \
      --condition dropout_0.3 --sequences 2 --fps 6
"""

import argparse
import os

import cv2
import numpy as np
import torch

from farmsim_perception.training.analyze import corrupt, predict
from farmsim_perception.data import NUM_CLASSES, Sequences
from farmsim_perception.engine import seq_generator
from farmsim_perception.training.evaluate import load

from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401
COLORS_BGR = np.array([(0, 0, 0), (40, 200, 40), (255, 90, 40), (0, 200, 255)], np.uint8)
TILE = (512, 384)  # (width, height) per panel
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay(image_bgr, class_ids, alpha=0.55):
    colour = cv2.resize(COLORS_BGR[class_ids], TILE, interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image_bgr, 1 - alpha, colour, alpha, 0)


def draw_line(tile, line, colour, thickness=3):
    h, w = tile.shape[:2]
    p0 = (int((line[0] + 1) * w / 2), int((line[1] + 1) * h / 2))
    p1 = (int((line[2] + 1) * w / 2), int((line[3] + 1) * h / 2))
    cv2.arrowedLine(tile, p0, p1, colour, thickness, tipLength=0.12)


def draw_box(tile, box, colour, thickness=2):
    h, w = tile.shape[:2]
    x0, y0, x1, y1 = box
    cv2.rectangle(tile, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), colour, thickness)


def caption(tile, title, lines=(), title_colour=(255, 255, 255)):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(tile, title, (8, 19), FONT, 0.6, title_colour, 2, cv2.LINE_AA)
    for i, text in enumerate(lines):
        y = tile.shape[0] - 10 - 22 * (len(lines) - 1 - i)
        cv2.putText(tile, text, (8, y), FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, text, (8, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def frame_iou(pred, gt):
    """Mean IoU over the classes present in either mask, for this frame alone."""
    ious = []
    for c in range(NUM_CLASSES):
        p, g = pred == c, gt == c
        union = (p | g).sum()
        if union:
            ious.append((p & g).sum() / union)
    return float(np.mean(ious)) if ious else float('nan')


def legend_strip(width):
    strip = np.zeros((34, width, 3), np.uint8)
    x = 10
    for name, colour in zip(('background', 'crop', 'corridor', 'bare bed'), COLORS_BGR):
        cv2.rectangle(strip, (x, 10), (x + 18, 26), tuple(int(v) for v in colour), -1)
        cv2.putText(strip, name, (x + 24, 24), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        x += 24 + 10 * len(name) + 24
    cv2.putText(strip, 'magenta arrow = predicted row line, yellow box = row bed', (x + 20, 24),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def render_sequence(seq, models, device, condition, writer, label):
    images = torch.from_numpy(seq['images']).to(device).permute(0, 3, 1, 2)
    x, hit = corrupt(images, condition, seq_generator(seq['name'] + condition, device))
    hit = hit.cpu().numpy()
    preds = {k: predict(m, x) for k, m in models.items()}
    names = list(models)
    running = {k: [] for k in names}

    for t in range(len(x)):
        inp = cv2.cvtColor((x[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inp = cv2.resize(inp, TILE, interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(seq['images'][t], TILE, interpolation=cv2.INTER_LINEAR)
        gt = overlay(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), seq['masks'][t])

        title = f'input  {condition}  |  {label}  frame {t + 1}/{len(x)}'
        top = [caption(inp.copy(), 'FRAME LOST' if hit[t] else title,
                       title_colour=(0, 160, 255) if hit[t] else (255, 255, 255)),
               caption(gt, 'ground truth')]

        bottom = []
        for k in names:
            p = preds[k]
            tile = overlay(inp.copy(), p['seg'][t])
            if p['bbox_prob'][t] > 0.5:
                draw_box(tile, p['bbox'][t], (0, 220, 255))
            draw_line(tile, p['line'][t], (255, 0, 255))
            running[k].append(frame_iou(p['seg'][t], seq['masks'][t]))
            lat_err = abs(p['pose'][t][0] - seq['pose'][t][0]) * 100
            bottom.append(caption(tile, k, (f'frame IoU {running[k][-1]:.2f}   mean {np.nanmean(running[k]):.2f}',
                                            f'lateral error {lat_err:5.1f} cm')))

        grid = np.vstack([np.hstack(top), np.hstack(bottom)])
        writer.write(np.vstack([grid, legend_strip(grid.shape[1])]))
    return {k: float(np.nanmean(v)) for k, v in running.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATASET))
    ap.add_argument('--runs', default=str(MODELS_DIR / 'row_net'))
    ap.add_argument('--models', nargs='+', default=['cnn', 'cnn_lstm'])
    ap.add_argument('--condition', default='dropout_0.3')
    ap.add_argument('--split', default='test')
    ap.add_argument('--sequences', type=int, default=2, help='how many test sequences to show')
    ap.add_argument('--fps', type=float, default=6.0, help='data is ~3 Hz; 6 fps plays at 2x')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Sequences(args.data, args.split, limit=args.sequences)
    models = {k: load(k, args.runs, device) for k in args.models}
    out = args.out or os.path.join(args.runs, f'video_{args.condition}.mp4')

    size = (TILE[0] * 2, TILE[1] * 2 + 34)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'could not open {out} for writing')
    for i, seq in enumerate(data.seqs, 1):
        means = render_sequence(seq, models, device, args.condition, writer, f'{seq["name"]}')
        print(f'[video] {i}/{len(data.seqs)} {seq["name"]}: '
              + ', '.join(f'{k} mean frame IoU {v:.3f}' for k, v in means.items()), flush=True)
    writer.release()
    print(f'[video] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
