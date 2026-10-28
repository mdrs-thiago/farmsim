"""
data.py

Loads dataset/row_perception_v1 from its training cache (resized uint8
images + semantic masks per sequence) plus annotations.jsonl, and serves either
single frames (CNN) or causal clips (CNN-LSTM). The cache is not stored in git
or copied with the dataset; a missing cache file is rebuilt from
sequences/<name>/images and masks/semantic on first use.

Images stay uint8 on the CPU; normalisation and augmentation run on the GPU
(`augment_batch`) so both models get exactly the same augmentation code path.
"""

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 4
CLASS_NAMES = ['background', 'crop', 'corridor', 'bare_bed']
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_cache(root, name, cache='256x192'):
    """Write cache/<cache>/<name>.npz from the full-resolution sequence."""
    size = tuple(int(v) for v in cache.split('x'))
    seq_dir = os.path.join(root, 'sequences', name)
    frames = sorted(os.listdir(os.path.join(seq_dir, 'images')))
    images, masks = [], []
    for f in frames:
        img = cv2.imread(os.path.join(seq_dir, 'images', f))
        sem = cv2.imread(os.path.join(seq_dir, 'masks', 'semantic', f.replace('.jpg', '.png')), -1)
        images.append(cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        masks.append(cv2.resize(sem, size, interpolation=cv2.INTER_NEAREST))
    out = os.path.join(root, 'cache', cache, f'{name}.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + '.tmp.npz', images=np.stack(images), masks=np.stack(masks))
    os.replace(out + '.tmp.npz', out)
    return out


def _targets(annotation_path, n_frames):
    """Per-frame regression targets as float32 arrays of length n_frames."""
    anns = [json.loads(line) for line in open(annotation_path)]
    assert len(anns) == n_frames, annotation_path
    bbox = np.zeros((n_frames, 4), np.float32)
    bbox_present = np.zeros(n_frames, np.float32)
    line = np.zeros((n_frames, 4), np.float32)
    line_vis = np.zeros((n_frames, 2), np.float32)
    pose = np.zeros((n_frames, 2), np.float32)
    for i, a in enumerate(anns):
        if a['row_bbox_xyxy'] is not None:
            x0, y0, x1, y1 = a['row_bbox_xyxy']
            bbox[i] = (x0 / 640, y0 / 480, x1 / 640, y1 / 480)
            bbox_present[i] = 1
        if a['row_line_norm'] is not None:
            line[i] = np.clip(a['row_line_norm'], -3, 3)
            line_vis[i] = a['row_line_visible']
        pose[i] = (a['lateral_offset_m'], a['heading_error_rad'])
    return {'bbox': bbox, 'bbox_present': bbox_present, 'line': line, 'line_vis': line_vis, 'pose': pose}


class Sequences:
    """All sequences of one split held in RAM."""

    def __init__(self, root, split, cache='256x192', limit=None, variant_root=None,
                 include_source_images=False):
        names = [n for n in open(os.path.join(root, 'splits', f'{split}.txt')).read().split() if n]
        if limit:
            names = names[:limit]
        self.names, self.seqs = names, []
        for n in names:
            cache_file = os.path.join(root, 'cache', cache, f'{n}.npz')
            if not os.path.exists(cache_file):
                build_cache(root, n, cache)
            npz = np.load(cache_file)
            images, masks = npz['images'], npz['masks']
            anomaly_masks = None
            source_images = None
            if variant_root is not None:
                source_images = images if include_source_images else None
                with np.load(os.path.join(variant_root, 'cache', cache, f'{n}.npz')) as variant:
                    images = variant['images']
                    anomaly_masks = variant['anomaly_masks']
                if anomaly_masks.shape != masks.shape or images.shape[:3] != masks.shape:
                    raise ValueError(f'{n}: anomaly variant is not aligned with source labels')
            t = _targets(os.path.join(root, 'sequences', n, 'annotations.jsonl'), len(images))
            seq = {'name': n, 'images': images, 'masks': masks, **t}
            if anomaly_masks is not None:
                seq['anomaly_masks'] = anomaly_masks
                if source_images is not None:
                    seq['source_images'] = source_images
            self.seqs.append(seq)

    def __len__(self):
        return len(self.seqs)

    @property
    def n_frames(self):
        return sum(len(s['images']) for s in self.seqs)


def _item(seq, idx):
    return {
        'image': torch.from_numpy(seq['images'][idx]).permute(*((0, 3, 1, 2) if np.ndim(idx) else (2, 0, 1))),
        'mask': torch.from_numpy(seq['masks'][idx].astype(np.int64)),
        **{k: torch.as_tensor(np.asarray(seq[k][idx])) for k in ('bbox', 'bbox_present', 'line', 'line_vis', 'pose')},
    }


class FrameDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences.seqs
        self.index = [(si, t) for si, s in enumerate(self.seqs) for t in range(len(s['images']))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, t = self.index[i]
        return _item(self.seqs[si], t)


class ClipDataset(Dataset):
    """Contiguous windows of `clip_len` frames. With `stride < clip_len` windows
    overlap; length is chosen so one epoch sees about as many frames as the
    FrameDataset epoch."""

    def __init__(self, sequences, clip_len=8, stride=None):
        self.seqs, self.clip_len = sequences.seqs, clip_len
        stride = stride or clip_len
        self.index = [(si, t) for si, s in enumerate(self.seqs)
                      for t in range(0, max(1, len(s['images']) - clip_len + 1), stride)
                      if len(s['images']) >= clip_len]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, t = self.index[i]
        return _item(self.seqs[si], np.arange(t, t + self.clip_len))


@torch.no_grad()
def augment_batch(images_uint8, generator=None, strength=1.0):
    """GPU augmentation for (B,3,H,W) or (B,T,3,H,W) uint8 images.

    Photometric jitter is shared across a clip (lighting does not change frame
    to frame); sensor noise, occluders and exposure spikes are drawn independently per frame,
    which is the kind of transient corruption a temporal model can see through.
    Returns float images in [0,1].
    """
    clip = images_uint8.dim() == 5
    x = images_uint8.float() / 255.0
    if not clip:
        x = x.unsqueeze(1)
    b, t, c, h, w = x.shape
    dev = x.device
    g = generator

    def rand(*shape):
        return torch.rand(*shape, device=dev, generator=g)

    # per-clip photometric jitter
    brightness = 1 + strength * 0.3 * (rand(b, 1, 1, 1, 1) * 2 - 1)
    contrast = 1 + strength * 0.3 * (rand(b, 1, 1, 1, 1) * 2 - 1)
    saturation = 1 + strength * 0.3 * (rand(b, 1, 1, 1, 1) * 2 - 1)
    x = x * brightness
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = x.mean(dim=2, keepdim=True)
    x = (x - gray) * saturation + gray

    # per-frame gaussian sensor noise
    sigma = strength * 0.06 * rand(b, t, 1, 1, 1)
    x = x + sigma * torch.randn(x.shape, device=dev, generator=g)

    # per-frame occluders (dust, leaves, glare): up to 2 random boxes, p=0.35 each
    for _ in range(2):
        on = rand(b, t) < 0.35 * strength
        bh = (rand(b, t) * 0.35 * h + 0.1 * h).long()
        bw = (rand(b, t) * 0.35 * w + 0.1 * w).long()
        y0 = (rand(b, t) * (h - bh)).long()
        x0 = (rand(b, t) * (w - bw)).long()
        yy = torch.arange(h, device=dev).view(1, 1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, 1, w)
        box = (on.view(b, t, 1, 1) & (yy >= y0.view(b, t, 1, 1)) & (yy < (y0 + bh).view(b, t, 1, 1))
               & (xx >= x0.view(b, t, 1, 1)) & (xx < (x0 + bw).view(b, t, 1, 1)))
        fill = rand(b, t, c, 1, 1)
        x = torch.where(box.unsqueeze(2), fill.expand(-1, -1, -1, h, w), x)

    # occasional heavy frame corruption (exposure spike / near-dropout), p=0.05
    drop = (rand(b, t, 1, 1, 1) < 0.05 * strength).float()
    x = x * (1 - drop) + drop * (0.15 * x + 0.85 * rand(b, t, 1, 1, 1))

    x = x.clamp(0, 1)
    return x if clip else x.squeeze(1)
