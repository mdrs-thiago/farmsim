"""
blend_check.py

Phase 7 validation: alpha-blend rendered label frames over their matching
real RGB frames (both keyed by the same frame_NNNNNN.png name), and check
that each label frame only contains the declared class colors for its
channel -- no antialiasing bleed. Void (black) is rendered fully
transparent so the underlying RGB shows through cleanly at frame edges /
outside the segmentation ROI.

Runs against render_labels.py's dual-channel output layout
(label_frames/vegetation/*_veg.png, label_frames/structure/*_struct.png)
-- one blend + one color-histogram check per channel per sampled frame.
Declared colors come straight from palettes.py (not a hardcoded set here)
so this can't silently drift out of sync if a palette ever changes.

Also computes and checks the FUSED channel (palettes.fuse_structure_vegetation)
on the fly from the vegetation + structure PNGs -- it isn't a separately
rendered file (see export_recording.py's docstring for why), so this is
the only place besides export_recording.py that materializes it; writing
it to --out-dir alongside the blend lets you eyeball the fused result
without running the full export.

Usage:
  python -m farmsim_perception.labeling.blend_check \
      --recording-dir recordings/phase7_test \
      --frames frame_000000 frame_000040 frame_000087 \
      --out-dir recordings/phase7_test/blends

With no --frames given, picks 5 evenly spaced frames from whatever's in
frames/ automatically (a reasonable spread for a quick near/mid/far/turn
check without having to know frame numbers in advance).
"""

import argparse
import os

import numpy as np
from PIL import Image

from farmsim_farm.palettes import (
    FUSED_PALETTE,
    STRUCT_PALETTE,
    VEG_PALETTE,
    classid_to_rgb,
    fuse_structure_vegetation,
    rgb_to_classid,
)

CHANNELS = [
    ('vegetation', 'veg', VEG_PALETTE),
    ('structure', 'struct', STRUCT_PALETTE),
]


def blend_one(rgb_path, label_path, out_path, alpha=130):
    rgb = Image.open(rgb_path).convert('RGBA')
    label = Image.open(label_path).convert('RGBA')
    arr = np.array(label)
    is_void = arr[:, :, :3].sum(axis=2) == 0
    arr[:, :, 3] = np.where(is_void, 0, alpha).astype('uint8')
    blended = Image.alpha_composite(rgb, Image.fromarray(arr))
    blended.convert('RGB').save(out_path)


def check_histogram(label_path, declared_palette):
    img = np.array(Image.open(label_path).convert('RGB'))
    colors = set(map(tuple, np.unique(img.reshape(-1, 3), axis=0).tolist()))
    # Palette tuples are BGR-labeled but render_labels.py writes files with
    # one channel-order reversal already applied (see its
    # _load_classid_raster docstring) -- so the ON-DISK colors are the
    # palette tuples reversed, once.
    declared_on_disk = {tuple(reversed(c)) for c in declared_palette}
    unexpected = colors - declared_on_disk
    return colors, unexpected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording-dir', required=True,
                         help='Directory with frames/ and label_frames/{vegetation,structure}/ subdirs '
                              '(record_camera_poses.py output_dir + render_labels.py --out-dir)')
    parser.add_argument('--frames', nargs='*', default=None,
                         help='Frame base names (no extension), e.g. frame_000040. '
                              'Default: 5 evenly spaced frames from frames/.')
    parser.add_argument('--out-dir', default=None,
                         help='Default: <recording-dir>/blends')
    parser.add_argument('--alpha', type=int, default=130)
    args = parser.parse_args()

    frames_dir = os.path.join(args.recording_dir, 'frames')
    out_dir = args.out_dir or os.path.join(args.recording_dir, 'blends')
    os.makedirs(out_dir, exist_ok=True)

    if args.frames:
        names = args.frames
    else:
        all_frames = sorted(f[:-4] for f in os.listdir(frames_dir) if f.endswith('.png'))
        if not all_frames:
            raise RuntimeError(f'No frames found in {frames_dir}')
        idx = np.linspace(0, len(all_frames) - 1, min(5, len(all_frames))).astype(int)
        names = [all_frames[i] for i in idx]
        print(f'No --frames given, picked {len(names)} evenly spaced: {names}')

    any_unexpected = False

    for name in names:
        rgb_path = os.path.join(frames_dir, name + '.png')
        if not os.path.exists(rgb_path):
            print(f'SKIP {name}: missing {rgb_path}')
            continue

        label_paths = {}
        for channel_dir, suffix, palette in CHANNELS:
            label_path = os.path.join(args.recording_dir, 'label_frames', channel_dir,
                                       f'{name}_{suffix}.png')
            if not os.path.exists(label_path):
                print(f'SKIP {name} [{channel_dir}]: missing {label_path}')
                continue
            label_paths[channel_dir] = label_path

            out_path = os.path.join(out_dir, f'{name}_{suffix}_blend.png')
            blend_one(rgb_path, label_path, out_path, args.alpha)

            colors, unexpected = check_histogram(label_path, palette)
            status = 'OK' if not unexpected else f'UNEXPECTED COLORS: {unexpected}'
            if unexpected:
                any_unexpected = True
            print(f'{name} [{channel_dir}]: wrote {out_path} -- {len(colors)} unique colors -- {status}')

        if 'vegetation' in label_paths and 'structure' in label_paths:
            veg_file = np.asarray(Image.open(label_paths['vegetation']).convert('RGB'))
            struct_file = np.asarray(Image.open(label_paths['structure']).convert('RGB'))
            veg_ids = rgb_to_classid(veg_file[:, :, ::-1], VEG_PALETTE)
            struct_ids = rgb_to_classid(struct_file[:, :, ::-1], STRUCT_PALETTE)
            fused_ids = fuse_structure_vegetation(struct_ids, veg_ids)
            fused_path = os.path.join(out_dir, f'{name}_fused.png')
            Image.fromarray(classid_to_rgb(fused_ids, FUSED_PALETTE)[:, :, ::-1], 'RGB').save(fused_path)

            out_path = os.path.join(out_dir, f'{name}_fused_blend.png')
            blend_one(rgb_path, fused_path, out_path, args.alpha)
            print(f'{name} [fused]: wrote {out_path}')

    if any_unexpected:
        print('\nFAIL: at least one frame has non-declared colors -- check texture '
              'filtering (must be NEAREST) and unlit/FLAT rendering.')
    else:
        print('\nAll checked frames: only declared class colors on both channels. '
              'Now look at the blended images -- does each label line up with the real image?')


if __name__ == '__main__':
    main()
