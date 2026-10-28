"""
visualize_dataset.py

Human-viewable QC visualization for an exported dataset session
(export_recording.py's output). The raw masks/{vegetation,structure,fused}
PNGs store per-pixel CLASS IDS (0, 1, 2, ...), not colors -- opened
directly they look almost solid black, since even the highest class id is
tiny next to the 0-255 range a normal image viewer expects. This colorizes
each mask via the same palettes.py the rest of the pipeline uses and
blends it over the matching RGB frame (from rgb.mp4), same 50%-alpha
convention as blend_check.py, so the class boundaries are actually visible
against the real image. Class 0 (void/background in every palette) is left
fully transparent, same as blend_check.py, so it doesn't tint frame edges
black.

Writes a playable review_<channel>.mp4 by default (every frame, RGB with
the mask overlaid) -- pass --out-frames-dir too if you'd rather step
through individual PNGs than scrub a video.

Usage:
  python -m farmsim_perception.labeling.visualize_dataset \
      --dataset-dir datasets/sim/vollebekk_farm_20250425__20250425/run_XXX \
      --channel fused
"""

import argparse
import json
import os

import cv2

from farmsim_farm.palettes import FUSED_PALETTE, STRUCT_PALETTE, VEG_PALETTE, classid_to_rgb

CHANNEL_PALETTES = {'vegetation': VEG_PALETTE, 'structure': STRUCT_PALETTE, 'fused': FUSED_PALETTE}


def blend_frame(rgb, class_ids, palette, alpha=0.5):
    """rgb, colored: BGR uint8 (cv2 convention, matching palette.py's own
    BGR-labeled tuples -- no channel-order conversion needed anywhere in
    this function). class_ids==0 (void/background in every palette here)
    is left untouched, not tinted, same as blend_check.py's blend_one."""
    colored = classid_to_rgb(class_ids, palette)
    is_background = class_ids == 0
    blended = cv2.addWeighted(rgb, 1 - alpha, colored, alpha, 0)
    out = rgb.copy()
    out[~is_background] = blended[~is_background]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', required=True, help='exported session dir (export_recording.py output)')
    parser.add_argument('--channel', choices=list(CHANNEL_PALETTES), default='fused')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--out-video', default=None, help='Default: <dataset-dir>/review_<channel>.mp4')
    parser.add_argument('--out-frames-dir', default=None,
                         help='Optional: also write individual blended PNGs here')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N frames')
    args = parser.parse_args()

    with open(os.path.join(args.dataset_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    mask_dir = os.path.join(args.dataset_dir, 'masks', args.channel)
    palette = CHANNEL_PALETTES[args.channel]

    cap = cv2.VideoCapture(os.path.join(args.dataset_dir, 'rgb.mp4'))
    out_video_path = args.out_video or os.path.join(args.dataset_dir, f'review_{args.channel}.mp4')
    if args.out_frames_dir:
        os.makedirs(args.out_frames_dir, exist_ok=True)

    n = manifest['frame_count'] if args.limit is None else min(args.limit, manifest['frame_count'])
    writer = None
    n_written = 0
    for i in range(n):
        ok, rgb = cap.read()
        if not ok:
            print(f'WARNING: rgb.mp4 ran out of frames at index {i}')
            break
        mask_path = os.path.join(mask_dir, f'frame_{i:06d}.png')
        class_ids = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if class_ids is None:
            print(f'WARNING: missing mask {mask_path}, skipping')
            continue

        blended = blend_frame(rgb, class_ids, palette, args.alpha)

        if writer is None:
            h, w = blended.shape[:2]
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                      manifest['nominal_fps'], (w, h))
        writer.write(blended)
        if args.out_frames_dir:
            cv2.imwrite(os.path.join(args.out_frames_dir, f'frame_{i:06d}.png'), blended)
        n_written += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f'Wrote {n_written} frame(s) to {out_video_path}')
    if args.out_frames_dir:
        print(f'Also wrote individual frames to {args.out_frames_dir}')


if __name__ == '__main__':
    main()
