#!/usr/bin/env python3
"""Create persistent, ground-anchored visual anomalies and exact edit masks.

The original images and row labels remain in --data. A variant cache contains
RGB images and one uint8 anomaly mask per frame (0=none, 1=puddle, 2=rut,
3=flat debris). The mask records pixels actually edited, at cache resolution.
"""

import argparse
import hashlib
import json
import os

import cv2
import numpy as np


from farmsim_utils.paths import DATASET_DIR, DEFAULT_DATASET, MODELS_DIR, OUTPUTS_DIR, RECORDINGS_DIR, REPO, field_dir  # noqa: F401
KINDS = ('puddle', 'rut', 'flat_debris')
COLORS_RGB = ((34, 48, 52), (64, 42, 29), (161, 153, 125))
ALPHAS = (0.68, 0.55, 0.82)
MASK_COLORS_RGB = np.array(((0, 0, 0), (30, 105, 145), (150, 65, 20), (225, 210, 100)), np.uint8)


def load_heightmap(path):
    with np.load(path) as d:
        return {k: d[k].copy() for k in ('heights', 'x0', 'y0', 'resolution')}


def terrain_z(heightmap, xy):
    h = heightmap['heights']
    x0, y0, res = (float(heightmap[k]) for k in ('x0', 'y0', 'resolution'))
    i = np.rint((xy[:, 0] - x0) / res).astype(int)
    j = np.rint((xy[:, 1] - y0) / res).astype(int)
    valid = (i >= 0) & (i < h.shape[0]) & (j >= 0) & (j < h.shape[1])
    z = np.full(len(xy), np.nan, np.float32)
    z[valid] = h[i[valid], j[valid]]
    return z


def make_anomalies(poses, heightmap, seed, count, semantic_masks=None, camera_info=None):
    """Place irregular polygons a few metres ahead of sampled camera poses.

    When labels and intrinsics are given, reject candidates that do not cover
    visible ground in their anchor frame. This prevents an 'active' session
    whose anomaly is entirely hidden by crop canopy or outside the image.
    """
    rng = np.random.default_rng(seed)
    eligible = np.arange(len(poses))
    if semantic_masks is not None:
        ground_pixels = np.isin(semantic_masks, (2, 3)).sum(axis=(1, 2))
        eligible = eligible[ground_pixels >= 100]
    if not len(eligible):
        return []
    anchors = eligible[np.linspace(0, len(eligible) - 1, count + 2, dtype=int)[1:-1]]
    anomalies = []
    for idx in anchors:
        t = np.asarray(poses[int(idx)]['T_world_optical'], dtype=float)
        forward = t[:2, 2].copy()
        norm = np.linalg.norm(forward)
        if norm < 0.2:
            continue
        forward /= norm
        sideways = np.array([-forward[1], forward[0]])
        for _ in range(16):
            center = t[:2, 3] + forward * rng.uniform(2.0, 4.5) + sideways * rng.uniform(-1.0, 1.0)
            kind = int(rng.integers(1, len(KINDS) + 1))
            if kind == 1:
                length, width = rng.uniform(0.5, 1.2), rng.uniform(0.3, 0.7)
            elif kind == 2:
                length, width = rng.uniform(1.0, 2.2), rng.uniform(0.10, 0.25)
            else:
                length, width = rng.uniform(0.22, 0.48), rng.uniform(0.18, 0.42)
            angle = rng.uniform(0, 2 * np.pi)
            along = np.cos(angle) * forward + np.sin(angle) * sideways
            across = np.array([-along[1], along[0]])
            theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            radial = rng.uniform(0.78, 1.13, len(theta))
            xy = center + np.outer(np.cos(theta) * radial * length, along) + np.outer(
                np.sin(theta) * radial * width, across)
            z = terrain_z(heightmap, xy)
            if not np.isfinite(z).all():
                continue
            vertices = np.column_stack((xy, z + 0.015))
            if semantic_masks is not None:
                ground = (semantic_masks[idx] == 2) | (semantic_masks[idx] == 3)
                h, w = ground.shape
                polygon = project_polygon(vertices, t, camera_info, w, h)
                if polygon is None:
                    continue
                footprint = np.zeros((h, w), np.uint8)
                cv2.fillPoly(footprint, [polygon], 1)
                if np.count_nonzero(footprint & ground) < 30:
                    continue
            anomalies.append({'id': len(anomalies) + 1, 'kind': kind,
                              'name': KINDS[kind - 1], 'vertices_world': vertices.tolist(),
                              'anchor_frame': int(idx)})
            break
    return anomalies


def project_polygon(vertices_world, T_world_optical, camera_info, width, height):
    t = np.asarray(T_world_optical, dtype=float)
    vertices = np.asarray(vertices_world, dtype=float)
    optical = (vertices - t[:3, 3]) @ t[:3, :3]
    if np.any(optical[:, 2] <= 0.10):
        return None
    sx = width / camera_info['width']
    sy = height / camera_info['height']
    u = sx * (camera_info['fx'] * optical[:, 0] / optical[:, 2] + camera_info['cx'])
    v = sy * (camera_info['fy'] * optical[:, 1] / optical[:, 2] + camera_info['cy'])
    if u.max() < 0 or v.max() < 0 or u.min() >= width or v.min() >= height:
        return None
    return np.rint(np.column_stack((u, v))).astype(np.int32)


def apply_anomalies(images, semantic_masks, poses, camera_info, anomalies):
    """Return edited RGB images and masks; only ground-class pixels are edited."""
    result = images.copy()
    masks = np.zeros(images.shape[:3], np.uint8)
    h, w = images.shape[1:3]
    for frame_idx, pose in enumerate(poses):
        ground = (semantic_masks[frame_idx] == 2) | (semantic_masks[frame_idx] == 3)
        if not ground.any():
            continue
        for anomaly in anomalies:
            polygon = project_polygon(anomaly['vertices_world'], pose['T_world_optical'],
                                      camera_info, w, h)
            if polygon is None:
                continue
            footprint = np.zeros((h, w), np.uint8)
            cv2.fillPoly(footprint, [polygon], 1)
            edited = (footprint != 0) & ground
            if not edited.any():
                continue
            kind = anomaly['kind']
            alpha = ALPHAS[kind - 1]
            color = np.asarray(COLORS_RGB[kind - 1], np.float32)
            original = result[frame_idx, edited].astype(np.float32)
            result[frame_idx, edited] = np.rint((1 - alpha) * original + alpha * color).astype(np.uint8)
            masks[frame_idx, edited] = kind
    return result, masks


def load_poses(recording_dir, annotations):
    path = os.path.join(recording_dir, 'poses.jsonl')
    poses_by_name = {p['frame']: p for p in (json.loads(line) for line in open(path))}
    return [poses_by_name[a['source_frame']] for a in annotations]


def write_preview(path, source, edited, masks):
    exposed = np.flatnonzero(np.any(masks != 0, axis=(1, 2)))
    pool = exposed if len(exposed) else np.arange(len(source))
    indices = pool[np.linspace(0, len(pool) - 1, min(4, len(pool)), dtype=int)]
    rows = [np.hstack((source[i], edited[i], MASK_COLORS_RGB[masks[i]])) for i in indices]
    preview = cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, preview)


def process_sequence(data_root, recordings_root, out_root, name, heightmap, seed, count,
                     affected_session_fraction, overwrite):
    out_cache = os.path.join(out_root, 'cache', '256x192', f'{name}.npz')
    out_seq = os.path.join(out_root, 'sequences', name)
    manifest_path = os.path.join(out_seq, 'anomalies.json')
    if os.path.exists(out_cache) or os.path.exists(manifest_path):
        if not overwrite:
            raise FileExistsError(f'{name}: variant already exists; pass --overwrite to replace it')
    cache_path = os.path.join(data_root, 'cache', '256x192', f'{name}.npz')
    with np.load(cache_path) as cache:
        images, semantic = cache['images'], cache['masks']
    seq_dir = os.path.join(data_root, 'sequences', name)
    annotations = [json.loads(line) for line in open(os.path.join(seq_dir, 'annotations.jsonl'))]
    recording_dir = os.path.join(recordings_root, name)
    poses = load_poses(recording_dir, annotations)
    camera_info = json.load(open(os.path.join(recording_dir, 'camera_info.json')))
    if len(images) != len(poses) or images.shape[:3] != semantic.shape:
        raise ValueError(f'{name}: images, masks and poses do not align')
    sequence_seed = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big')
    active = np.random.default_rng(sequence_seed).random() < affected_session_fraction
    anomalies = make_anomalies(poses, heightmap, sequence_seed, count if active else 0,
                              semantic_masks=semantic, camera_info=camera_info)
    edited, masks = apply_anomalies(images, semantic, poses, camera_info, anomalies)

    os.makedirs(os.path.dirname(out_cache), exist_ok=True)
    os.makedirs(os.path.join(out_seq, 'masks', 'anomaly'), exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(os.path.join(out_seq, 'masks', 'anomaly', f'{i:06d}.png'), mask)
    np.savez_compressed(out_cache, images=edited, anomaly_masks=masks)
    write_preview(os.path.join(out_seq, 'preview.jpg'), images, edited, masks)
    manifest = {'source_dataset': os.path.abspath(data_root), 'source_recording': os.path.abspath(recording_dir),
                'seed': seed, 'sequence_seed': sequence_seed, 'requested_anomalies': count,
                'affected_session_fraction': affected_session_fraction,
                'classes': {'0': 'none',
                **{str(i + 1): n for i, n in enumerate(KINDS)}}, 'anomalies': anomalies,
                'frame_affected_pixels': (masks != 0).sum(axis=(1, 2)).tolist(),
                'frame_count': len(images), 'resolution': [images.shape[2], images.shape[1]]}
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'{name}: {len(anomalies)} anomalies, {(masks != 0).any(axis=(1, 2)).sum()}/{len(images)} affected frames',
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=str(DEFAULT_DATASET))
    ap.add_argument('--recordings', default=str(RECORDINGS_DIR))
    ap.add_argument('--heightmap', default=str(field_dir(20250425) / 'terrain_heightmap.npz'))
    ap.add_argument('--out', default=str(DATASET_DIR / 'row_anomalies_v1'))
    ap.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    ap.add_argument('--limit-sequences', type=int, default=None)
    ap.add_argument('--anomalies-per-sequence', type=int, default=1)
    ap.add_argument('--affected-session-fraction', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if os.path.realpath(args.out) == os.path.realpath(args.data):
        ap.error('--out must differ from --data')
    if args.anomalies_per_sequence < 0:
        ap.error('--anomalies-per-sequence must be nonnegative')
    if not 0 <= args.affected_session_fraction <= 1:
        ap.error('--affected-session-fraction must be between 0 and 1')
    heightmap = load_heightmap(args.heightmap)
    names = open(os.path.join(args.data, 'splits', f'{args.split}.txt')).read().split()
    if args.limit_sequences is not None:
        names = names[:args.limit_sequences]
    for name in names:
        process_sequence(args.data, args.recordings, args.out, name, heightmap,
                         args.seed, args.anomalies_per_sequence,
                         args.affected_session_fraction, args.overwrite)


if __name__ == '__main__':
    main()
