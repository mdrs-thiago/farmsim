#!/usr/bin/env python3
"""Render a side-by-side QC video of a session: raw RGB | fused-mask overlay |
row bbox/polygon | row line.  Usage: make_label_video.py SESSION_DIR [-o OUT.mp4]"""
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np

# fused class ids -> BGR (background, crop, row corridor, bare bed)
PALETTE = np.array([(0, 0, 0), (0, 255, 0), (255, 0, 0), (0, 220, 255)], np.uint8)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', type=Path)
    ap.add_argument('-o', '--out', type=Path)
    ap.add_argument('--alpha', type=float, default=0.45)
    a = ap.parse_args()
    m = json.loads((a.session / 'manifest.json').read_text())
    out = a.out or a.session / 'qc_side_by_side.mp4'
    fps = m.get('nominal_fps', 3.0)
    cap = cv2.VideoCapture(str(a.session / 'rgb.mp4'))
    tmp = Path(tempfile.mkdtemp())
    raw_path = tmp / 'raw.mp4'
    writer = None
    for i, fr in enumerate(m['per_frame']):
        ok, rgb = cap.read()
        if not ok:
            break
        h, w = rgb.shape[:2]
        mask = cv2.imread(str(a.session / 'masks' / 'fused' / f'frame_{i:06d}.png'), 0)
        seg = rgb.copy()
        if mask is not None:
            color = PALETTE[np.clip(mask, 0, 3)]
            fg = mask > 0
            seg[fg] = cv2.addWeighted(rgb, 1 - a.alpha, color, a.alpha, 0)[fg]
        box = rgb.copy()
        if fr.get('row_bbox'):
            x0, y0, x1, y1 = fr['row_bbox']
            cv2.rectangle(box, (x0, y0), (x1, y1), (0, 255, 255), 2)
        if fr.get('row_polygon'):
            cv2.polylines(box, [np.int32(fr['row_polygon'])], True, (255, 0, 255), 2)
        line = rgb.copy()
        if fr.get('row_line_px'):
            # Extend the 1 m -> 5 m label segment across the ground: from the
            # image bottom up to the labelled horizon (top of non-void mask).
            x0, y0, x1, y1 = fr['row_line_px']
            if abs(y1 - y0) > 1e-6:
                y_top = y1
                if mask is not None and (mask > 0).any():
                    y_top = min(y1, float(np.nonzero((mask > 0).any(axis=1))[0][0]))
                x_at = lambda y: x0 + (x1 - x0) * (y - y0) / (y1 - y0)
                pb = (int(round(x_at(h - 1))), h - 1)
                pt = (int(round(x_at(y_top))), int(round(y_top)))
                cv2.arrowedLine(line, pb, pt, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.04)
            ang = fr.get('row_line_angle_deg')
            if ang is not None:
                cv2.putText(line, f'{ang:+.1f} deg', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles = [label(rgb, f'raw  #{i}'), label(seg, 'segmentation (fused)'),
                 label(box, 'row bbox / polygon'), label(line, 'row line')]
        frame = np.hstack(tiles)
        if writer is None:
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, frame.shape[1::-1])
        writer.write(frame)
    writer.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw_path), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], check=True)
    else:
        shutil.move(raw_path, out)
    print(f'wrote {out} ({i + 1} frames @ {fps:.2f} fps)')


if __name__ == '__main__':
    main()
