"""Sequence-preserving dataset for CNN versus CNN-LSTM line estimation.

Reads the row_perception_v1 layout (sequences/<name>/images/*.jpg plus
annotations.jsonl), the same data the row_net models train on. The official
splits/<split>.txt files decide train/val/test when they exist; the hash split
is the fallback for a folder of sequences without split files.
"""

import hashlib
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def find_sessions(dataset_root):
    seq_root = os.path.join(dataset_root, 'sequences')
    return [os.path.join(seq_root, name) for name in sorted(os.listdir(seq_root))
            if os.path.isfile(os.path.join(seq_root, name, 'annotations.jsonl'))]


def official_splits(dataset_root, sessions):
    """(train, val, test) from splits/*.txt, or None if the dataset has none."""
    split_dir = os.path.join(dataset_root, 'splits')
    if not os.path.isdir(split_dir):
        return None
    by_name = {os.path.basename(s): s for s in sessions}
    out = []
    for split in ('train', 'val', 'test'):
        names = open(os.path.join(split_dir, f'{split}.txt')).read().split()
        out.append([by_name[n] for n in names if n in by_name])
    return tuple(out)


def split_sessions(sessions, val_fraction=0.15, test_fraction=0.15, seed=0):
    """Stable independent hash assignment; adding data cannot reshuffle old runs."""
    out = {'train': [], 'val': [], 'test': []}
    for session in sessions:
        key = f'{seed}:{os.path.basename(session)}'.encode()
        score = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        if score < test_fraction:
            out['test'].append(session)
        elif score < test_fraction + val_fraction:
            out['val'].append(session)
        else:
            out['train'].append(session)
    return out['train'], out['val'], out['test']


class LineSequenceDataset(Dataset):
    """One item is a contiguous fixed-length clip from exactly one session."""

    def __init__(self, session_dirs, sequence_length=8, stride=8,
                 image_size=(96, 72), preload=True, cache_dir=None, augment=False):
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sequences = []
        self.clips = []
        self.skipped_sessions = []
        self.augment = augment
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        for session_dir in session_dirs:
            name = os.path.basename(session_dir)
            images_dir = os.path.join(session_dir, 'images')
            image_names = sorted(os.listdir(images_dir)) if os.path.isdir(images_dir) else []
            annotations = [json.loads(line) for line in open(os.path.join(session_dir, 'annotations.jsonl'))]
            if len(image_names) != len(annotations):
                self.skipped_sessions.append(name)
                continue
            frames = []
            for index, (image_name, ann) in enumerate(zip(image_names, annotations)):
                line = ann.get('row_line_norm')
                visible = ann.get('row_line_visible') or [False, False]
                frames.append({
                    'image_path': os.path.join(images_dir, image_name),
                    'line': line if line is not None else [0.0] * 4,
                    'visible': visible if line is not None else [False, False],
                    'session': name,
                    'frame_index': index,
                })
            if len(frames) < sequence_length:
                self.skipped_sessions.append(name)
                continue
            if preload:
                cache_path = (os.path.join(cache_dir, f'{name}_{image_size[0]}x{image_size[1]}.npy')
                              if cache_dir else None)
                images = None
                if cache_path and os.path.isfile(cache_path):
                    candidate = np.load(cache_path)
                    if candidate.shape == (len(frames), image_size[1], image_size[0], 3):
                        images = candidate
                if images is None:
                    images = np.empty((len(frames), image_size[1], image_size[0], 3), dtype=np.uint8)
                    for i, frame in enumerate(frames):
                        image = cv2.imread(frame['image_path'])
                        if image is None:
                            raise RuntimeError(f"Could not decode {frame['image_path']}")
                        image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
                        images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if cache_path:
                        temporary = cache_path + '.tmp.npy'
                        np.save(temporary, images)
                        os.replace(temporary, cache_path)
                for frame, image in zip(frames, images):
                    frame['image'] = image
            seq_index = len(self.sequences)
            self.sequences.append(frames)
            starts = list(range(0, len(frames) - sequence_length + 1, stride))
            last = len(frames) - sequence_length
            if starts[-1] != last:
                starts.append(last)
            self.clips.extend((seq_index, start) for start in starts)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        seq_index, start = self.clips[index]
        frames = self.sequences[seq_index][start:start + self.sequence_length]
        width, height = self.image_size
        images = []
        for frame in frames:
            image = frame.get('image')
            if image is None:
                image = cv2.imread(frame['image_path'])
                if image is None:
                    raise RuntimeError(f"Could not decode {frame['image_path']}")
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            images.append(torch.from_numpy(image).permute(2, 0, 1))
        images = torch.stack(images)
        line = torch.tensor([f['line'] for f in frames], dtype=torch.float32)
        if self.augment:
            # Apply geometric changes consistently to the entire clip; varying
            # them frame-by-frame would manufacture artificial temporal jitter.
            if torch.rand(()) < 0.5:
                images = images.flip(-1)
                line[:, (0, 2)] *= -1.0
            gain = 0.85 + 0.30 * torch.rand(())
            bias = -0.04 + 0.08 * torch.rand(())
            images = (images * gain + bias).clamp(0.0, 1.0)
        return {
            'images': images,
            'line': line,
            'visible': torch.tensor([f['visible'] for f in frames], dtype=torch.bool),
            'session': frames[0]['session'],
            'frame_index': torch.tensor([f['frame_index'] for f in frames]),
        }

    def visibility_counts(self):
        # Counts unique source frames, not duplicated overlapping clip entries.
        near = far = both = total = 0
        for sequence in self.sequences:
            for frame in sequence:
                n, f = frame['visible']
                near += int(n); far += int(f); both += int(n and f); total += 1
        return {'frames': total, 'near': near, 'far': far, 'both': both}
