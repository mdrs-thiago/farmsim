#!/usr/bin/env python3
"""
build_dataset.py

Consolidates the exported sim sessions (datasets/sim/<world>__<date>/<session>/)
plus their raw recordings (recordings/<session>/frames/) into one organised,
self-describing dataset for row perception:

  <out>/
    README.md                     class definitions, label semantics, statistics
    dataset.json                  machine-readable version of the above
    sessions.csv                  one row per sequence: split, strategy, quality, frames, ...
    splits/{train,val,test}.txt   sequence names (session-level, grouped by world seed)
    coco/{split}_row.json         row-bed detection in COCO format (bbox + polygon)
    sequences/<session>/
      images/NNNNNN.jpg           RGB 640x480
      masks/semantic/NNNNNN.png   4-class id mask (see SEMANTIC_CLASSES)
      masks/structure/NNNNNN.png  original 3-class structure ids
      masks/vegetation/NNNNNN.png original 3-class vegetation ids
      annotations.jsonl           per-frame labels, same order as images
      meta.json                   run parameters of the recording
    cache/<W>x<H>/<session>.npz   resized uint8 images + semantic masks for training

Semantic mask. The exported masks/fused channel never contains its ROW class:
the vegetation renderer marks corridor pixels as VOID and fuse_structure_vegetation
requires both channels to be observed, so corridors collapse into background.
Here the corridor is taken from the structure channel alone:
  corridor  <- structure == CORRIDOR
  crop      <- structure == BED and vegetation == VEG
  bare_bed  <- structure == BED and vegetation != VEG
  background otherwise (sky, outside the surveyed footprint)

Labels. With --labels-root, structure/vegetation masks are read from a fresh
render_labels.py output (<labels-root>/<session>/{structure,vegetation}/
<frame>_{struct,veg}.png) instead of the exported masks/. Most exported sessions
were rendered before farm/fields/20250425/structure_raster.png was rebuilt from the
annotated plot polygons (2026-09-07 12:37), and their structure masks miss almost
all corridors. Sessions without a fresh render are left out.

Splits. Session names are grouped by world seed (a few seeds were re-recorded, and
the same seed gives the same generated world), stratified by
(control strategy, measured trajectory quality), then allocated 70/15/15 by a
stable hash so adding sessions never moves existing ones between splits.

Usage:
  python tools/row_perception/build_dataset.py \
      --out datasets/thorvald_row_perception_v1 --workers 10
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401
from farmsim_farm import palettes  # noqa: E402

SEMANTIC_CLASSES = ['background', 'crop', 'corridor', 'bare_bed']
SEM_BACKGROUND, SEM_CROP, SEM_CORRIDOR, SEM_BARE_BED = range(4)
SEMANTIC_COLORS_RGB = [(0, 0, 0), (40, 200, 40), (40, 90, 255), (255, 200, 0)]
SPLIT_FRACTIONS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
CACHE_SIZE = (256, 192)


def semantic_from_channels(struct, veg):
    sem = np.full(struct.shape, SEM_BACKGROUND, np.uint8)
    bed = struct == palettes.STRUCT_BED
    sem[bed & (veg == palettes.VEG_VEG)] = SEM_CROP
    sem[bed & (veg != palettes.VEG_VEG)] = SEM_BARE_BED
    sem[struct == palettes.STRUCT_CORRIDOR] = SEM_CORRIDOR
    return sem


def read_rgb_frames(session, session_dir, per_frame):
    """Prefer the lossless PNGs from recordings/; fall back to decoding rgb.mp4."""
    frames_dir = str(RECORDINGS_DIR / session / 'frames')
    if os.path.isdir(frames_dir) and all(
            os.path.exists(os.path.join(frames_dir, pf['frame'])) for pf in per_frame):
        for pf in per_frame:
            yield cv2.imread(os.path.join(frames_dir, pf['frame'])), 'png'
        return
    cap = cv2.VideoCapture(os.path.join(session_dir, 'rgb.mp4'))
    for _ in per_frame:
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f'{session}: rgb.mp4 has fewer frames than the manifest')
        yield img, 'mp4'


def frame_annotation(i, pf):
    line_norm = pf.get('row_line_norm')
    return {
        'index': i,
        'image': f'images/{i:06d}.jpg',
        'mask': f'masks/semantic/{i:06d}.png',
        'stamp': pf['stamp'],
        'source_frame': pf['frame'],
        'row_id': pf.get('row_id'),
        # detection: the crop bed the robot straddles, pixel coords in 640x480
        'row_bbox_xyxy': pf.get('row_bbox'),
        'row_polygon': pf.get('row_polygon'),
        # row centre line: near (1 m) and far (5 m) look-ahead points
        'row_line_px': pf.get('row_line_px'),
        'row_line_norm': line_norm,
        'row_line_visible': pf.get('row_line_in_frame') if line_norm is not None else [False, False],
        'lateral_offset_m': pf.get('lateral_offset'),
        'heading_error_rad': pf.get('heading_error'),
        'lateral_offset_rows': pf.get('lateral_offset_rows'),
        'cmd_vel': pf.get('cmd_vel'),
    }


def label_paths(session_dir, labels_root, pf):
    session = os.path.basename(session_dir)
    if labels_root:
        stem = pf['frame'][:-4]
        return (os.path.join(labels_root, session, 'structure', f'{stem}_struct.png'),
                os.path.join(labels_root, session, 'vegetation', f'{stem}_veg.png'))
    idx = pf['frame_index']
    return (os.path.join(session_dir, 'masks', 'structure', f'frame_{idx:06d}.png'),
            os.path.join(session_dir, 'masks', 'vegetation', f'frame_{idx:06d}.png'))


def build_session(args):
    session_dir, out_root, overwrite, labels_root = args
    session = os.path.basename(session_dir)
    manifest = json.load(open(os.path.join(session_dir, 'manifest.json')))
    per_frame = manifest['per_frame']
    seq_dir = os.path.join(out_root, 'sequences', session)
    cache_path = os.path.join(out_root, 'cache', f'{CACHE_SIZE[0]}x{CACHE_SIZE[1]}', f'{session}.npz')
    done_flag = os.path.join(seq_dir, '.complete')
    class_px = np.zeros(len(SEMANTIC_CLASSES), np.int64)

    if os.path.exists(done_flag) and os.path.exists(cache_path) and not overwrite:
        stats = json.load(open(done_flag))
        return session, stats

    for sub in ('images', 'masks/semantic', 'masks/structure', 'masks/vegetation'):
        os.makedirs(os.path.join(seq_dir, sub), exist_ok=True)

    small_imgs, small_masks, annotations = [], [], []
    source = None
    for i, (pf, (img, source)) in enumerate(zip(per_frame, read_rgb_frames(session, session_dir, per_frame))):
        struct_path, veg_path = label_paths(session_dir, labels_root, pf)
        struct, veg = cv2.imread(struct_path, -1), cv2.imread(veg_path, -1)
        if img is None or struct is None or veg is None:
            raise RuntimeError(f'{session}: missing data for frame {pf["frame"]}')
        # render_labels.py writes palette colours (BGR); exported masks are already class ids
        if struct.ndim == 3:
            struct = palettes.rgb_to_classid(struct, palettes.STRUCT_PALETTE)
        if veg.ndim == 3:
            veg = palettes.rgb_to_classid(veg, palettes.VEG_PALETTE)
        sem = semantic_from_channels(struct, veg)
        class_px += np.bincount(sem.ravel(), minlength=len(SEMANTIC_CLASSES))

        cv2.imwrite(os.path.join(seq_dir, 'images', f'{i:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(seq_dir, 'masks', 'semantic', f'{i:06d}.png'), sem)
        cv2.imwrite(os.path.join(seq_dir, 'masks', 'structure', f'{i:06d}.png'), struct)
        cv2.imwrite(os.path.join(seq_dir, 'masks', 'vegetation', f'{i:06d}.png'), veg)
        annotations.append(frame_annotation(i, pf))

        small_imgs.append(cv2.cvtColor(cv2.resize(img, CACHE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        small_masks.append(cv2.resize(sem, CACHE_SIZE, interpolation=cv2.INTER_NEAREST))

    with open(os.path.join(seq_dir, 'annotations.jsonl'), 'w') as f:
        for a in annotations:
            f.write(json.dumps(a) + '\n')
    meta = {k: v for k, v in manifest.items() if k != 'per_frame'}
    meta['rgb_source'] = source
    json.dump(meta, open(os.path.join(seq_dir, 'meta.json'), 'w'), indent=2)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, images=np.stack(small_imgs), masks=np.stack(small_masks))

    stats = {'frames': len(annotations), 'class_px': class_px.tolist(), 'rgb_source': source}
    json.dump(stats, open(done_flag, 'w'))
    return session, stats


def stable_score(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / 2**64


def assign_splits(sessions, quality):
    """Group by world seed, stratify groups, allocate by hash rank within stratum."""
    groups = defaultdict(list)
    for s in sessions:
        groups[re.search(r'seed(\d+)', s).group(1)].append(s)
    strata = defaultdict(list)
    for seed, members in groups.items():
        q = quality.get(members[0], {})
        strata[(q.get('strategy', '?'), q.get('measured_quality', '?'))].append(seed)
    split_of = {}
    for key, seeds in strata.items():
        seeds.sort(key=lambda sd: stable_score(f'row_perception_v1:{sd}'))
        n = len(seeds)
        n_test = round(n * SPLIT_FRACTIONS['test'])
        n_val = round(n * SPLIT_FRACTIONS['val'])
        if n >= 3:
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        for rank, seed in enumerate(seeds):
            split = 'test' if rank < n_test else 'val' if rank < n_test + n_val else 'train'
            for s in groups[seed]:
                split_of[s] = split
    return split_of


def coco_for_split(out_root, sessions):
    images, anns = [], []
    for s in sessions:
        for line in open(os.path.join(out_root, 'sequences', s, 'annotations.jsonl')):
            a = json.loads(line)
            image_id = len(images) + 1
            images.append({'id': image_id, 'file_name': f'sequences/{s}/{a["image"]}',
                           'width': 640, 'height': 480, 'sequence': s, 'frame_index': a['index']})
            if a['row_bbox_xyxy'] is None:
                continue
            x0, y0, x1, y1 = a['row_bbox_xyxy']
            poly = [c for p in (a['row_polygon'] or []) for c in p]
            anns.append({'id': len(anns) + 1, 'image_id': image_id, 'category_id': 1,
                         'bbox': [x0, y0, x1 - x0, y1 - y0], 'area': (x1 - x0) * (y1 - y0),
                         'segmentation': [poly] if poly else [], 'iscrowd': 0})
    return {'info': {'description': 'Thorvald row perception v1 - current crop bed'},
            'categories': [{'id': 1, 'name': 'row_bed'}], 'images': images, 'annotations': anns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(OUTPUTS_DIR / 'sim_export' / 'vollebekk_farm_20250425__20250425'))
    ap.add_argument('--out', default=str(DEFAULT_DATASET))
    ap.add_argument('--quality-csv', default=str(DEFAULT_DATASET / 'trajectory_quality_report.csv'))
    ap.add_argument('--labels-root', default=None,
                    help='fresh render_labels.py output per session; sessions without <session>/.done are skipped')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    session_dirs = sorted(os.path.join(args.src, d) for d in os.listdir(args.src)
                          if os.path.isfile(os.path.join(args.src, d, 'manifest.json')))
    if args.labels_root:
        missing = [d for d in session_dirs if not os.path.exists(os.path.join(args.labels_root, os.path.basename(d), '.done'))]
        for d in missing:
            print(f'[build] no fresh labels, skipping {os.path.basename(d)}')
        session_dirs = [d for d in session_dirs if d not in missing]
    quality = {r['session']: r for r in csv.DictReader(open(args.quality_csv))}
    os.makedirs(args.out, exist_ok=True)

    stats = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (session, st) in enumerate(pool.map(build_session, [(d, args.out, args.overwrite, args.labels_root) for d in session_dirs]), 1):
            stats[session] = st
            if n % 25 == 0 or n == len(session_dirs):
                print(f'[build] {n}/{len(session_dirs)} sessions', flush=True)

    sessions = sorted(stats)
    split_of = assign_splits(sessions, quality)
    os.makedirs(os.path.join(args.out, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'coco'), exist_ok=True)
    summary = {}
    for split in SPLIT_FRACTIONS:
        members = [s for s in sessions if split_of[s] == split]
        open(os.path.join(args.out, 'splits', f'{split}.txt'), 'w').write('\n'.join(members) + '\n')
        coco = coco_for_split(args.out, members)
        json.dump(coco, open(os.path.join(args.out, 'coco', f'{split}_row.json'), 'w'))
        px = np.sum([stats[s]['class_px'] for s in members], axis=0)
        summary[split] = {
            'sequences': len(members),
            'frames': int(sum(stats[s]['frames'] for s in members)),
            'frames_with_row_bbox': len(coco['annotations']),
            'class_pixel_fraction': dict(zip(SEMANTIC_CLASSES, (px / px.sum()).round(4).tolist())),
            'strata': dict(Counter(f"{quality.get(s, {}).get('strategy', '?')}/{quality.get(s, {}).get('measured_quality', '?')}"
                                   for s in members)),
        }

    with open(os.path.join(args.out, 'sessions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session', 'split', 'seed', 'frames', 'strategy', 'measured_quality', 'rgb_source'])
        for s in sessions:
            q = quality.get(s, {})
            w.writerow([s, split_of[s], re.search(r'seed(\d+)', s).group(1), stats[s]['frames'],
                        q.get('strategy', '?'), q.get('measured_quality', '?'), stats[s]['rgb_source']])

    info = {
        'name': 'thorvald_row_perception_v1',
        'source': os.path.relpath(args.src, REPO),
        'image_size': [640, 480],
        'nominal_fps': 3.03,
        'semantic_classes': SEMANTIC_CLASSES,
        'semantic_colors_rgb': SEMANTIC_COLORS_RGB,
        'cache_size': list(CACHE_SIZE),
        'splits': summary,
    }
    json.dump(info, open(os.path.join(args.out, 'dataset.json'), 'w'), indent=2)
    write_readme(args.out, info)
    print(json.dumps(summary, indent=2))


def write_readme(out, info):
    rows = '\n'.join(
        f"| {k} | {v['sequences']} | {v['frames']} | {v['frames_with_row_bbox']} | "
        + ' / '.join(f"{c} {100 * p:.1f}%" for c, p in v['class_pixel_fraction'].items()) + ' |'
        for k, v in info['splits'].items())
    classes = '\n'.join(f'| {i} | {c} | {info["semantic_colors_rgb"][i]} |' for i, c in enumerate(info['semantic_classes']))
    text = f"""# Thorvald row perception dataset (v1)

Simulated Thorvald II camera sequences from the Vollebekk farm world (survey 2025-04-25),
organised for semantic segmentation, row-bed detection and row-line / pose regression,
with the temporal order kept so sequence models can use it.

Built by `tools/row_perception/build_dataset.py` from `{info['source']}` and `recordings/`.

## Splits (session-level, grouped by world seed, stratified by controller + measured quality)

| split | sequences | frames | frames with row bbox | semantic pixel share |
|---|---|---|---|---|
{rows}

## Layout

```
sequences/<session>/images/NNNNNN.jpg        RGB 640x480, ~3 Hz
sequences/<session>/masks/semantic/NNNNNN.png class ids below
sequences/<session>/masks/structure|vegetation/ original channel ids (tools/label_pipeline/palettes.py)
sequences/<session>/annotations.jsonl         one JSON object per frame, same order as images
sequences/<session>/meta.json                 run parameters (camera pose, controller, weeds, ...)
splits/<split>.txt                            sequence names
coco/<split>_row.json                         COCO detection, category 1 = row_bed (bbox + polygon)
cache/{info['cache_size'][0]}x{info['cache_size'][1]}/<session>.npz  uint8 images (T,H,W,3 RGB) + masks (T,H,W)
sessions.csv                                  per-sequence split / controller / quality
```

## Semantic classes

| id | class | RGB colour |
|---|---|---|
{classes}

`corridor` comes from the structure channel. The exported `masks/fused` never contains
its row class, because the vegetation channel marks corridors as void.

All masks were re-rendered with `tools/label_pipeline/render_labels.py` from the plot-polygon
structure raster of 2026-09-07. Most exported sessions had been rendered from the older
relief-based raster, which misses almost all corridors. Three sessions have no source recording
to re-render from (`run_20260907_121131/121308/121435`), so they are excluded.

## Per-frame annotation fields

- `row_bbox_xyxy`, `row_polygon`: the crop bed the robot straddles, in pixels. Null when it is not visible.
- `row_line_px` / `row_line_norm`: the row centre line at 1 m (near) and 5 m (far) ahead, as
  `[u_near, v_near, u_far, v_far]`. Normalised coordinates run from -1 to 1. `row_line_visible` gives
  in-frame flags for the near and far points.
- `lateral_offset_m`, `heading_error_rad`: robot base_link relative to the row centre line.
- `cmd_vel`: the controller command at that frame.
"""
    open(os.path.join(out, 'README.md'), 'w').write(text)


if __name__ == '__main__':
    main()
