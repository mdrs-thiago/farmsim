#!/usr/bin/env python3
"""Decode and objectively assess every exported RGB video.

Checks container/frame integrity, timing metadata, exposure, sharpness,
duplicate/frozen frames, inter-frame visual change, and sampled MP4-vs-source
PNG PSNR.  The source comparison distinguishes a poor recording from damage
introduced by video encoding.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def finite_mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


def analyze_session(session_dir, recordings_root, source_samples):
    manifest = json.loads((session_dir / 'manifest.json').read_text())
    pf = manifest.get('per_frame', [])
    expected = int(manifest.get('frame_count', len(pf)))
    video_path = session_dir / 'rgb.mp4'
    result = {
        'session': session_dir.name, 'expected_frames': expected,
        'video_exists': video_path.is_file(), 'decoded_frames': 0,
    }
    if not video_path.is_file():
        result['issues'] = 'missing_video'
        return result

    sample_indices = set(np.linspace(0, max(expected - 1, 0),
                                     min(source_samples, expected), dtype=int).tolist())
    rec_frames = recordings_root / session_dir.name / 'frames'
    cap = cv2.VideoCapture(str(video_path))
    result['container_frames'] = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    result['video_fps'] = float(cap.get(cv2.CAP_PROP_FPS))
    result['width'] = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    result['height'] = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    luma, clipped_dark, clipped_bright, sharpness = [], [], [], []
    deltas, psnrs = [], []
    previous = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        luma.append(float(gray.mean()))
        clipped_dark.append(float(np.mean(gray <= 5)))
        clipped_bright.append(float(np.mean(gray >= 250)))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous is not None:
            deltas.append(float(np.mean(cv2.absdiff(small, previous))))
        previous = small

        if i in sample_indices and i < len(pf):
            source = cv2.imread(str(rec_frames / pf[i]['frame']))
            if source is not None and source.shape == frame.shape:
                mse = float(np.mean((source.astype(np.float32) - frame.astype(np.float32)) ** 2))
                psnrs.append(99.0 if mse == 0.0 else 10.0 * math.log10(255.0 ** 2 / mse))
        i += 1
    cap.release()

    stamps = [f.get('stamp') for f in pf]
    stamp_deltas = [b - a for a, b in zip(stamps, stamps[1:])
                    if a is not None and b is not None and b > a]
    pose_ages = [abs(float(f['pose_age_s'])) for f in pf if f.get('pose_age_s') is not None]
    result.update({
        'decoded_frames': i,
        'manifest_fps': float(manifest.get('nominal_fps', 0.0)),
        'stamp_dt_median_s': percentile(stamp_deltas, 50),
        'stamp_dt_p95_s': percentile(stamp_deltas, 95),
        'pose_age_p95_s': percentile(pose_ages, 95),
        'luma_mean': finite_mean(luma),
        'luma_p05': percentile(luma, 5),
        'luma_p95': percentile(luma, 95),
        'dark_clipped_pixel_fraction': finite_mean(clipped_dark),
        'bright_clipped_pixel_fraction': finite_mean(clipped_bright),
        'sharpness_laplacian_median': percentile(sharpness, 50),
        'sharpness_laplacian_p05': percentile(sharpness, 5),
        'frame_delta_median': percentile(deltas, 50),
        'frame_delta_p95': percentile(deltas, 95),
        'frozen_pair_fraction': (float(np.mean(np.asarray(deltas) < 0.2)) if deltas else None),
        'source_psnr_mean_db': finite_mean(psnrs),
        'source_psnr_min_db': min(psnrs) if psnrs else None,
        'source_samples_compared': len(psnrs),
    })

    issues = []
    if i != expected or result['container_frames'] != expected:
        issues.append('frame_count_mismatch')
    if result['width'] != 640 or result['height'] != 480:
        issues.append('unexpected_resolution')
    mfps = result['manifest_fps']
    if mfps and abs(result['video_fps'] - mfps) / mfps > 0.02:
        issues.append('fps_mismatch')
    if result['frozen_pair_fraction'] is not None and result['frozen_pair_fraction'] > 0.01:
        issues.append('frozen_frames')
    if result['source_psnr_mean_db'] is not None and result['source_psnr_mean_db'] < 30.0:
        issues.append('low_encoding_psnr')
    if result['dark_clipped_pixel_fraction'] is not None and result['dark_clipped_pixel_fraction'] > 0.10:
        issues.append('dark_clipping')
    if result['bright_clipped_pixel_fraction'] is not None and result['bright_clipped_pixel_fraction'] > 0.10:
        issues.append('bright_clipping')
    result['issues'] = ','.join(issues)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root', default='datasets/sim/vollebekk_farm_20250425__20250425')
    ap.add_argument('--recordings-root', default='outputs/recordings')
    ap.add_argument('--source-samples', type=int, default=10,
                    help='evenly spaced MP4 frames compared to original PNG per session')
    ap.add_argument('--csv', help='write per-session metrics')
    ap.add_argument('--json', help='write aggregate summary and per-session metrics')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    sessions = sorted(p.parent for p in root.glob('*/manifest.json'))
    rows = []
    for n, session in enumerate(sessions, 1):
        rows.append(analyze_session(session, Path(args.recordings_root), args.source_samples))
        if n % 50 == 0:
            print(f'  decoded {n}/{len(sessions)} sessions', flush=True)

    numeric_keys = (
        'luma_mean', 'dark_clipped_pixel_fraction', 'bright_clipped_pixel_fraction',
        'sharpness_laplacian_median', 'frame_delta_median', 'frame_delta_p95',
        'frozen_pair_fraction', 'source_psnr_mean_db', 'pose_age_p95_s',
    )
    summary = {
        'sessions': len(rows),
        'expected_frames': sum(r['expected_frames'] for r in rows),
        'decoded_frames': sum(r['decoded_frames'] for r in rows),
        'sessions_with_issues': sum(bool(r.get('issues')) for r in rows),
        'issue_counts': {},
        'metric_percentiles_across_sessions': {},
    }
    for row in rows:
        for issue in filter(None, row.get('issues', '').split(',')):
            summary['issue_counts'][issue] = summary['issue_counts'].get(issue, 0) + 1
    for key in numeric_keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        summary['metric_percentiles_across_sessions'][key] = {
            'p05': percentile(values, 5), 'median': percentile(values, 50),
            'p95': percentile(values, 95),
        }

    print(json.dumps(summary, indent=2))
    worst = sorted((r for r in rows if r.get('issues')), key=lambda r: r['session'])
    if worst:
        print('Flagged sessions:')
        for row in worst:
            print(f'  {row["session"]}: {row["issues"]}')

    if args.csv and rows:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'summary': summary, 'sessions': rows}, f, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
