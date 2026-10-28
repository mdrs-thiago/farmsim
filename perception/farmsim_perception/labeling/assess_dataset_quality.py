#!/usr/bin/env python3
"""
assess_dataset_quality.py

Per-session tracking quality over the exported dataset, so the 85/15
small-deviation/erratic split can be checked against what the robot ACTUALLY
did rather than against the label generate_run_params.py intended.

This matters because the sessions recorded before the row-direction fix were
driven by a controller that periodically resolved a crop row backwards (14.5%
of frames measured over a 59-session sample), steering away from the row. The
labels for those sessions are correct once repaired, but the trajectories are
not what `trajectory_quality` claims -- a run tagged "small deviation" may
have bounced. This script reports the measured spread so those sessions can be
kept as deliberate diversity, down-weighted, or re-recorded.

Reads datasets/sim/<world>__<date>/<session>/manifest.json only -- no images,
no bags, so it is fast over hundreds of sessions.

Usage:
  python3 tools/assess_dataset_quality.py
  python3 tools/assess_dataset_quality.py --dataset-root datasets/sim --csv out.csv
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

# |lateral_offset| p90 below this metre threshold, and |heading_error| p90
# below the degree threshold, counts as "tracked the row". Chosen from the
# measured behaviour of the strategies that were verified stable: pd came in
# at 0.15 m / 4.8 deg and pure_pursuit at 0.19 m / 6.2 deg mean, so these
# thresholds sit comfortably above good tracking without admitting a run that
# wandered a full row spacing (~1.5 m).
GOOD_LATERAL_P90_M = 0.45
GOOD_HEADING_P90_DEG = 25.0


def session_stats(manifest_path):
    m = json.load(open(manifest_path))
    pf = m.get('per_frame', [])
    lat = np.array([f['lateral_offset'] for f in pf if f.get('lateral_offset') is not None])
    head = np.array([f['heading_error'] for f in pf if f.get('heading_error') is not None])
    if lat.size == 0 or head.size == 0:
        return None

    rp = m.get('run_params') or {}
    rid = [f.get('row_id') for f in pf if f.get('row_id') is not None]
    switches = sum(1 for a, b in zip(rid, rid[1:]) if a != b)

    head_deg = np.degrees(np.abs(head))
    lat_abs = np.abs(lat)
    lat_p90 = float(np.percentile(lat_abs, 90))
    head_p90 = float(np.percentile(head_deg, 90))
    tail_start = len(lat_abs) // 2
    tail_lat_p90 = float(np.percentile(lat_abs[tail_start:], 90))
    tail_head_p90 = float(np.percentile(head_deg[tail_start:], 90))
    angular = [abs(float(f['cmd_vel']['angular_z'])) for f in pf
               if f.get('cmd_vel') is not None]

    return {
        'session': m.get('session_name', os.path.basename(os.path.dirname(manifest_path))),
        'strategy': rp.get('control_strategy', '?'),
        'intended_quality': rp.get('trajectory_quality', '?'),
        'frames': len(pf),
        'lat_mean': float(lat_abs.mean()),
        'initial_lat': float(lat_abs[0]),
        'lat_p90': lat_p90,
        'lat_max': float(lat_abs.max()),
        'head_mean_deg': float(head_deg.mean()),
        'head_p90_deg': head_p90,
        'tail_lat_p90': tail_lat_p90,
        'tail_head_p90_deg': tail_head_p90,
        'cmd_angular_abs_mean': float(np.mean(angular)) if angular else None,
        'row_switches': switches,
        'measured_quality': ('small_deviation'
                             if lat_p90 <= GOOD_LATERAL_P90_M and head_p90 <= GOOD_HEADING_P90_DEG
                             else 'erratic'),
        'measured_quality_tail': ('small_deviation'
                                  if tail_lat_p90 <= GOOD_LATERAL_P90_M and
                                  tail_head_p90 <= GOOD_HEADING_P90_DEG
                                  else 'erratic'),
        'navigation_profile': rp.get('navigation_profile', 'legacy/diverse'),
        'camera_height': rp.get('camera_height'),
        'camera_pitch': rp.get('camera_pitch'),
        'camera_x': rp.get('camera_x'),
        'n_weeds': rp.get('n_weeds'),
        'extrinsics_repaired': bool(rp.get('_extrinsics_repaired', False)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', default='datasets/sim')
    ap.add_argument('--csv', help='also write the per-session table here')
    args = ap.parse_args()

    manifests = sorted(glob.glob(os.path.join(args.dataset_root, '*', '*', 'manifest.json')))
    rows = [s for s in (session_stats(p) for p in manifests) if s]
    if not rows:
        print(f'No manifests found under {args.dataset_root}')
        return

    n = len(rows)
    measured = defaultdict(int)
    measured_tail = defaultdict(int)
    intended = defaultdict(int)
    mismatch = 0
    for r in rows:
        measured[r['measured_quality']] += 1
        measured_tail[r['measured_quality_tail']] += 1
        intended[r['intended_quality']] += 1
        if r['intended_quality'] in ('small_deviation', 'erratic') and \
                r['intended_quality'] != r['measured_quality']:
            mismatch += 1

    print(f'{n} exported session(s) under {args.dataset_root}')
    print(f'  repaired extrinsics: {sum(r["extrinsics_repaired"] for r in rows)}/{n}')

    print('\nIntended trajectory_quality (what generate_run_params.py asked for):')
    for k, v in sorted(intended.items(), key=lambda kv: -kv[1]):
        print(f'  {k:16s} {v:4d}  ({100*v/n:.1f}%)')

    print(f'\nMeasured trajectory quality (|lat| p90 <= {GOOD_LATERAL_P90_M} m '
          f'and |heading| p90 <= {GOOD_HEADING_P90_DEG} deg):')
    for k, v in sorted(measured.items(), key=lambda kv: -kv[1]):
        print(f'  {k:16s} {v:4d}  ({100*v/n:.1f}%)')
    print(f'  sessions whose measured quality disagrees with the intended one: '
          f'{mismatch} ({100*mismatch/n:.1f}%)')
    print('  tail-half only (separates persistent instability from spawn recovery):')
    for k, v in sorted(measured_tail.items(), key=lambda kv: -kv[1]):
        print(f'    {k:16s} {v:4d}  ({100*v/n:.1f}%)')

    print('\nPer control strategy:')
    print(f'  {"strategy":14s} {"n":>4s} {"|lat| mean":>11s} {"|lat| p90":>10s} '
          f'{"|head| mean":>12s} {"row switches":>13s} {"good%":>7s}')
    by_strat = defaultdict(list)
    for r in rows:
        by_strat[r['strategy']].append(r)
    for strat, rs in sorted(by_strat.items(), key=lambda kv: -len(kv[1])):
        good = sum(1 for r in rs if r['measured_quality'] == 'small_deviation')
        print(f'  {strat:14s} {len(rs):4d} '
              f'{np.mean([r["lat_mean"] for r in rs]):10.3f}m '
              f'{np.mean([r["lat_p90"] for r in rs]):9.3f}m '
              f'{np.mean([r["head_mean_deg"] for r in rs]):11.2f}d '
              f'{np.mean([r["row_switches"] for r in rs]):13.2f} '
              f'{100*good/len(rs):6.1f}%  '
              f'tail-good={100*np.mean([r["measured_quality_tail"] == "small_deviation" for r in rs]):5.1f}%')

    initial_bins = ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, float('inf')))
    print('\nTracking pass rate by initial |lateral offset|:')
    for lo, hi in initial_bins:
        rs = [r for r in rows if lo <= r['initial_lat'] < hi]
        if not rs:
            continue
        label = f'{lo:.2f}-{hi:.2f}m' if math.isfinite(hi) else f'>={lo:.2f}m'
        all_good = np.mean([r['measured_quality'] == 'small_deviation' for r in rs])
        tail_good = np.mean([r['measured_quality_tail'] == 'small_deviation' for r in rs])
        print(f'  {label:12s} n={len(rs):3d}  whole={100*all_good:5.1f}%  tail={100*tail_good:5.1f}%')

    worst = sorted(rows, key=lambda r: -r['lat_p90'])[:10]
    print('\n10 worst-tracking sessions (by |lateral_offset| p90):')
    for r in worst:
        print(f'  {r["session"]:38s} {r["strategy"]:13s} intended={r["intended_quality"]:15s} '
              f'lat_p90={r["lat_p90"]:.3f}m head_p90={r["head_p90_deg"]:6.1f}d '
              f'switches={r["row_switches"]}')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nWrote {args.csv}')


if __name__ == '__main__':
    main()
