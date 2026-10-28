#!/usr/bin/env python3
"""Render an overview map and before/after/mask examples for an anomaly variant."""

import argparse
import glob
import json
import os

import cv2
import numpy as np
import yaml

from farmsim_perception.anomalies.make_anomaly_variant import KINDS, MASK_COLORS_RGB
from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, field_dir


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def project_topdown(points, meta):
    pose = meta['camera_pose']
    r_wc = rotation_matrix(pose['roll'], pose['pitch'], pose['yaw'])
    camera_xyz = np.array([pose['x'], pose['y'], pose['z']])
    sensor = (np.asarray(points) - camera_xyz) @ r_wc
    x_opt, y_opt, z_opt = -sensor[:, 1], -sensor[:, 2], sensor[:, 0]
    intr = meta['intrinsics']
    u = intr['fx'] * x_opt / z_opt + intr['cx']
    v = intr['fy'] * y_opt / z_opt + intr['cy']
    return np.column_stack((u, v))


def label(image, text, x, y, scale=0.72, color=(255, 255, 255)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 4,
                cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
                cv2.LINE_AA)


def make_map(manifests, topdown_path, meta, out_path):
    background = cv2.imread(topdown_path)
    if background is None:
        raise FileNotFoundError(topdown_path)
    polygons = []
    for _, manifest in manifests:
        for anomaly in manifest['anomalies']:
            polygons.append((anomaly['kind'], project_topdown(anomaly['vertices_world'], meta)))
    if not polygons:
        raise ValueError('variant has no visible anomalies to map')
    all_pixels = np.vstack([xy for _, xy in polygons])
    xmin, ymin = np.floor(all_pixels.min(axis=0) - 250).astype(int)
    xmax, ymax = np.ceil(all_pixels.max(axis=0) + 250).astype(int)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(background.shape[1], xmax), min(background.shape[0], ymax)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError('anomaly polygons do not project into the top-down image')
    crop = background[ymin:ymax, xmin:xmax]
    target_w = 1500
    scale = target_w / crop.shape[1]
    target_h = int(round(crop.shape[0] * scale))
    map_image = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    overlay = map_image.copy()
    for kind, xy in polygons:
        poly = np.rint((xy - [xmin, ymin]) * scale).astype(np.int32)
        color = tuple(int(v) for v in MASK_COLORS_RGB[kind][::-1])
        cv2.fillPoly(overlay, [poly], color)
    map_image = cv2.addWeighted(overlay, 0.65, map_image, 0.35, 0)
    for kind, xy in polygons:
        poly = np.rint((xy - [xmin, ymin]) * scale).astype(np.int32)
        color = tuple(int(v) for v in MASK_COLORS_RGB[kind][::-1])
        cv2.polylines(map_image, [poly], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(map_image, [poly], True, color, 2, cv2.LINE_AA)
        center = poly.mean(axis=0).astype(int)
        cv2.circle(map_image, tuple(center), 5, (255, 255, 255), -1, cv2.LINE_AA)

    header = np.full((94, target_w, 3), (31, 35, 38), np.uint8)
    label(header, f'Anomaly locations across {len(manifests)} test sequences', 20, 33, 0.85)
    counts = [sum(a['kind'] == i for _, m in manifests for a in m['anomalies'])
              for i in (1, 2, 3)]
    x = 25
    for kind, (name, count) in enumerate(zip(KINDS, counts), 1):
        color = tuple(int(v) for v in MASK_COLORS_RGB[kind][::-1])
        cv2.rectangle(header, (x, 58), (x + 18, 76), color, -1)
        label(header, f'{name.replace("_", " ")}: {count}', x + 28, 75, 0.58)
        x += 350
    label(header, 'World-anchored image edits', 1080, 74, 0.43)
    cv2.imwrite(out_path, np.vstack((header, map_image)), [cv2.IMWRITE_JPEG_QUALITY, 93])
    return counts


def make_examples(manifests, source_root, variant_root, out_path):
    rows = []
    for kind, name in enumerate(KINDS, 1):
        candidates = [(session, manifest) for session, manifest in manifests
                      if any(a['kind'] == kind for a in manifest['anomalies'])]
        if not candidates:
            continue
        session, manifest = max(candidates, key=lambda item: max(item[1]['frame_affected_pixels']))
        with np.load(os.path.join(source_root, 'cache', '256x192', f'{session}.npz')) as src:
            source = src['images']
        with np.load(os.path.join(variant_root, 'cache', '256x192', f'{session}.npz')) as var:
            edited, masks = var['images'], var['anomaly_masks']
        areas = (masks == kind).sum(axis=(1, 2))
        frame = int(areas.argmax())
        source_bgr = cv2.cvtColor(source[frame], cv2.COLOR_RGB2BGR)
        edited_bgr = cv2.cvtColor(edited[frame], cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(MASK_COLORS_RGB[masks[frame]], cv2.COLOR_RGB2BGR)
        tiles = [cv2.resize(im, (512, 384), interpolation=cv2.INTER_NEAREST)
                 for im in (source_bgr, edited_bgr, mask_bgr)]
        strip = np.hstack(tiles)
        label(strip, f'{name.replace("_", " ")}  |  {session}  frame {frame}', 16, 34, 0.64)
        rows.append(strip)
    header = np.full((62, 1536, 3), (31, 35, 38), np.uint8)
    for x, text in ((14, 'Source image'), (526, 'With anomaly'), (1038, 'Anomaly mask')):
        label(header, text, x, 42, 0.9)
    cv2.imwrite(out_path, np.vstack([header, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 94])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', default=str(DATASET_DIR / 'row_anomalies_v1'))
    ap.add_argument('--source', default=str(DEFAULT_DATASET))
    ap.add_argument('--topdown', default=str(field_dir(20250425) / 'topdown.png'))
    ap.add_argument('--topdown-meta', default=str(field_dir(20250425) / 'topdown.yaml'))
    args = ap.parse_args()
    manifests = [(os.path.basename(os.path.dirname(path)), json.load(open(path)))
                 for path in glob.glob(os.path.join(args.variant, 'sequences', '*', 'anomalies.json'))]
    manifests.sort(key=lambda item: item[0])
    if not manifests:
        ap.error(f'no anomaly manifests found in {args.variant}')
    meta = yaml.safe_load(open(args.topdown_meta))
    map_path = os.path.join(args.variant, 'map_overview.jpg')
    examples_path = os.path.join(args.variant, 'examples.jpg')
    counts = make_map(manifests, args.topdown, meta, map_path)
    make_examples(manifests, args.source, args.variant, examples_path)
    print(f'{map_path}: {counts[0]} puddles, {counts[1]} ruts, {counts[2]} flat debris')
    print(examples_path)


if __name__ == '__main__':
    main()
