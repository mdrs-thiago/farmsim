#!/usr/bin/env python3
"""
run_perception.py -- run only the perception module.

offline (default): the CNN over recorded dataset sequences, in frame order,
  scored against the ground truth (lateral offset, heading, row line) with an
  overlay video per sequence.
live: the robot drives a path on ground truth while the CNN runs on the live
  camera; its estimate is logged next to the truth (and shown when
  sim.headless=false).

  python run_perception.py
  python run_perception.py perception.checkpoint=models/row_net/cnn_lstm/last.pt offline.split=val
  python run_perception.py --live sim.headless=false

Output: outputs/perception/<name>/
"""

import json
import sys

import cv2
import numpy as np
import yaml

from farmsim_utils.config import parse
from farmsim_utils.paths import OUTPUTS_DIR
from farmsim_utils.pipeline import repo_path, timestamp

SEG_COLORS = np.array([(0, 0, 0), (40, 200, 40), (40, 90, 255), (255, 200, 0)], np.uint8)


def overlay(rgb, est, gt):
    h, w = rgb.shape[:2]
    seg = cv2.resize(SEG_COLORS[est['seg']], (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.addWeighted(rgb, 0.65, seg, 0.35, 0)

    def line(vals, color):
        if vals is None:
            return
        p = [(int((vals[i] + 1) * w / 2), int((vals[i + 1] + 1) * h / 2)) for i in (0, 2)]
        cv2.line(img, p[0], p[1], color, 2)
    line(gt.get('row_line_norm'), (255, 255, 255))
    line([est['u_near'], est['v_near'], est['u_far'], est['v_far']], (255, 0, 0))
    txt = (f"e {est['lateral_offset']:+.2f}/{gt['lateral_offset_m']:+.2f} m  "
           f"psi {np.degrees(est['heading_error']):+.1f}/{np.degrees(gt['heading_error_rad']):+.1f} deg  "
           f"conf {est['confidence']:.2f}")
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def offline(cfg, out_dir):
    from farmsim_perception.inference import RowPerception
    oc = cfg['offline']
    root = repo_path(oc['dataset'])
    names = (root / 'splits' / f"{oc['split']}.txt").read_text().split()
    if oc.get('limit'):
        names = names[:oc['limit']]
    perception = RowPerception(repo_path(cfg['perception']['checkpoint']), cfg['perception']['device'])
    summary = {}
    for name in names:
        seq = root / 'sequences' / name
        anns = [json.loads(line) for line in open(seq / 'annotations.jsonl')]
        frames = sorted((seq / 'images').iterdir())
        perception.reset()
        writer = None
        err_e, err_psi = [], []
        for f, gt in zip(frames, anns):
            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            est = perception(rgb)
            err_e.append(abs(est['lateral_offset'] - gt['lateral_offset_m']))
            err_psi.append(abs(est['heading_error'] - gt['heading_error_rad']))
            if oc.get('video'):
                img = overlay(rgb, est, gt)
                if writer is None:
                    writer = cv2.VideoWriter(str(out_dir / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             6, (img.shape[1], img.shape[0]))
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if writer:
            writer.release()
        summary[name] = {'frames': len(anns), 'mae_offset_m': float(np.mean(err_e)),
                         'mae_heading_deg': float(np.degrees(np.mean(err_psi)))}
        print(f"{name}: {summary[name]}")
    allf = sum(s['frames'] for s in summary.values())
    summary['_mean'] = {k: float(sum(s[k] * s['frames'] for s in summary.values()) / allf)
                        for k in ('mae_offset_m', 'mae_heading_deg')}
    print('mean', summary['_mean'])
    return summary


def live(cfg, out_dir):
    from run_control import run
    cfg = dict(cfg, run={'path': cfg['live']['path'], 'pose_source': 'gt', 'perception': True,
                         'checkpoint': cfg['perception']['checkpoint'],
                         'max_duration_s': cfg['live']['max_duration_s']})
    return run(cfg, out_dir)


def main():
    cfg, args = parse(__doc__, 'run_perception.yaml',
                      add_args=lambda ap: ap.add_argument('--live', action='store_true'))
    out = OUTPUTS_DIR / 'perception' / f"{timestamp()}_{'live' if args.live else 'offline'}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    result = live(cfg, out) if args.live else offline(cfg, out)
    with open(out / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'results in {out}')


if __name__ == '__main__':
    sys.exit(main())
