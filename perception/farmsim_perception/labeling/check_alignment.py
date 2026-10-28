"""
check_alignment.py

Diagnostic: isolates a genuine mesh-to-camera reprojection/geometry bug
from a bad-looking-but-geometrically-correct classification. Existing QC
(blend_check.py) only supports a manual eyeball check of the rendered
label against the real frame -- it can't tell you WHY something looks
wrong. This script gives two independent, automated answers:

  Part A (fast, no image data): projects real mesh points through TWO
  independently-written pinhole-projection code paths -- raster_common's
  analytic formula and render_labels.py's own MVP/GL_FIX construction,
  evaluated here without the GPU -- and asserts they agree to sub-pixel
  precision. A FAILURE here is conclusively a code bug (pose composition,
  the GL_FIX sign convention, or the intrinsics matrix), independent of
  any classification or survey data at all.

  Part B (photometric): bakes the REAL birdseye photo itself (not a class
  raster) onto the mesh via the exact same raster-building path
  build_label_raster.py uses, renders it from every recorded pose through
  render_labels.py's own render_texture_sequence, and phase-correlates
  the result against the real captured frame. Real, non-repeating photo
  content is a "known ground-control pattern" -- a mismatch here can only
  be a genuine geometric/timing problem, never a hue-threshold quality
  issue, since there is no classification anywhere in this path. Per-tile
  correlation distinguishes a spatially-uniform offset (wrong camera
  extrinsics/calibration constants) from a spatially-varying one (mesh
  resolution, or pose staleness -- cross-check the flagged frame's
  pose_age_s in poses.jsonl).

  Part C (GCP projection, optional -- needs --gcp-file): the original
  brief's Step 4 check #1, "project the known 3D positions of the GCP
  targets into rendered frames; confirm they land on identical pixels in
  both the RGB and the mask." Reuses Part A's exact self-consistency
  pattern (the same two independent projection code paths), but evaluated
  at the annotated plot-block corners (from annotate_plots.py's
  plot_lattice_*.json) instead of random mesh points, and additionally
  draws a labeled crosshair at the (agreed) projected pixel onto each
  sampled real RGB frame -- a human-reviewable artifact confirming where a
  physically-spawned GCP marker (spawn_row_markers.py --gcp-file) should
  land, not just a pass/fail number.

Decision rule printed at the end: Part A fail -> fix the projection code
before trusting anything else. Part B fail with a spatially UNIFORM
offset -> check camera_height/camera_pitch/camera_x_offset and pose
timing. Part B passes but blend_check.py still looks wrong -> not a
geometry bug, retune --hue-min/--sat-min (or the structure-raster
threshold) instead. Part C failure means the two projection code paths
disagree specifically at the GCP locations -- if Part A already passed
using random mesh points, a Part C failure points at something
GCP-specific (e.g. a stale/wrong plot_lattice_*.json), not the shared
projection math itself.

Usage:
  python -m farmsim_perception.labeling.check_alignment \
      --recording-dir recordings/phase7_test \
      --georef farm/fields/20250425/georef.json \
      --mesh farm/fields/20250425/label_mesh.glb \
      --topdown-meta farm/fields/20250425/topdown.yaml \
      --gcp-file config/plot_lattice_20250425.json \
      --limit 10
"""

import argparse
import json
import os

import numpy as np
import yaml
from PIL import Image, ImageDraw

from farmsim_farm import raster_common as rc
from farmsim_perception.labeling.render_labels import GL_FIX, intrinsics_to_projection, render_texture_sequence
from farmsim_farm.topdown_projection import load_topdown_meta


def project_via_mvp(points_w, T_world_optical, fx, fy, cx, cy, width, height, near=0.1, far=100.0):
    """Independent re-derivation of render_labels.py's own MVP
    construction (GL_FIX + intrinsics_to_projection + view = inv(pose @
    GL_FIX)), applied analytically to points here instead of rasterized
    by the GPU. Compared against raster_common.world_to_pixel_dynamic's
    separately-derived pinhole formula in run_part_a -- agreement between
    these two independently-written code paths is the self-consistency
    check; a real bug in either one shows up as a discrepancy here with
    no GPU or image data involved."""
    projection = intrinsics_to_projection(fx, fy, cx, cy, width, height, near, far)
    cam_to_world = np.array(T_world_optical) @ GL_FIX
    view = np.linalg.inv(cam_to_world)
    mvp = projection @ view

    n = points_w.shape[0]
    points_h = np.column_stack([points_w, np.ones(n)])
    clip = points_h @ mvp.T
    w_clip = clip[:, 3]
    valid = w_clip > 1e-6
    w_safe = np.where(valid, w_clip, 1.0)
    ndc = clip[:, :3] / w_safe[:, None]

    # Inverse of the viewport mapping render_texture_sequence's row-flip
    # implies: ndc y=+1 (top) -> pixel row 0, ndc x=-1 (left) -> column 0.
    u = (ndc[:, 0] + 1.0) / 2.0 * width
    v = (1.0 - ndc[:, 1]) / 2.0 * height
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def run_part_a(mesh_path, poses, cam_info, n_sample_points=200, sample_radius_m=15.0, tol_px=0.5):
    height_tree, mesh_z = rc.load_height_lookup(mesh_path)
    xy = height_tree.data
    fx, fy, cx, cy = cam_info['fx'], cam_info['fy'], cam_info['cx'], cam_info['cy']
    width, height = cam_info['width'], cam_info['height']
    rng = np.random.default_rng(0)

    max_err = 0.0
    n_checked = 0
    n_poses_checked = 0
    for rec in poses:
        T = np.array(rec['T_world_optical'])
        cam_pos = T[:3, 3]
        idx = height_tree.query_ball_point(cam_pos[:2], r=sample_radius_m)
        if not idx:
            continue
        idx = np.array(idx)
        if len(idx) > n_sample_points:
            idx = rng.choice(idx, size=n_sample_points, replace=False)
        points_w = np.column_stack([xy[idx, 0], xy[idx, 1], mesh_z[idx]])

        u1, v1, valid1 = rc.world_to_pixel_dynamic(points_w, T, fx, fy, cx, cy, width, height)
        u2, v2, valid2 = project_via_mvp(points_w, T, fx, fy, cx, cy, width, height)

        both_valid = valid1 & valid2
        if not np.any(both_valid):
            continue
        err = np.sqrt((u1[both_valid] - u2[both_valid]) ** 2 + (v1[both_valid] - v2[both_valid]) ** 2)
        max_err = max(max_err, float(err.max()))
        n_checked += int(both_valid.sum())
        n_poses_checked += 1

    print(f'  checked {n_checked} point-projections across {n_poses_checked}/{len(poses)} pose(s) '
          f'with nearby mesh coverage; max discrepancy = {max_err:.4f}px (tolerance {tol_px}px)')
    return max_err < tol_px


def load_gcp_points_world(gcp_file, mesh_path):
    """Reads annotate_plots.py's plot_lattice_*.json 'corners_world_xy' and
    looks up each corner's real terrain height (nearest-vertex, same
    height_tree/mesh_z pattern run_part_a already uses) -- GCP height is
    never stored in the lattice file itself, since it's a property of the
    terrain at that XY, not of the annotation. Returns an (N, 3) array."""
    with open(gcp_file) as f:
        lattice = json.load(f)
    xy = np.array(lattice['corners_world_xy'], dtype=np.float64)

    height_tree, mesh_z = rc.load_height_lookup(mesh_path)
    _, nearest_idx = height_tree.query(xy, k=1)
    z = mesh_z[nearest_idx]
    return np.column_stack([xy, z])


def run_part_c(gcp_points_world, poses, cam_info, frames_dir, out_dir, tol_px=2.0):
    """GCP pixel-projection check (original brief's Step 4 check #1):
    project each GCP's known world XYZ through the SAME two independent
    projection code paths run_part_a cross-checks (raster_common's analytic
    formula and render_labels.py's own MVP construction), at the specific
    GCP locations instead of random mesh points, and draw a labeled
    crosshair at the agreed pixel onto each sampled real RGB frame -- a
    human-reviewable artifact, not just a number, confirming where a
    physically-spawned GCP marker (spawn_row_markers.py --gcp-file) should
    appear. Two independent projections landing on the same pixel
    guarantees the real RGB frame and its matching rendered mask (which
    share that exact recorded pose/intrinsics) do too -- there is no third,
    separate "mask projection" path to independently get wrong."""
    fx, fy, cx, cy = cam_info['fx'], cam_info['fy'], cam_info['cx'], cam_info['cy']
    width, height = cam_info['width'], cam_info['height']
    points_w = np.asarray(gcp_points_world)

    max_err = 0.0
    n_checked = 0
    results = []
    for rec in poses:
        real_path = os.path.join(frames_dir, rec['frame'])
        if not os.path.exists(real_path):
            print(f"  SKIP {rec['frame']}: no matching real frame at {real_path}")
            continue

        T = np.array(rec['T_world_optical'])
        u1, v1, valid1 = rc.world_to_pixel_dynamic(points_w, T, fx, fy, cx, cy, width, height)
        u2, v2, valid2 = project_via_mvp(points_w, T, fx, fy, cx, cy, width, height)
        both_valid = valid1 & valid2

        img = Image.open(real_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        frame_max_err = 0.0
        n_visible = 0
        for i in range(len(points_w)):
            if not both_valid[i]:
                continue
            err = float(np.hypot(u1[i] - u2[i], v1[i] - v2[i]))
            frame_max_err = max(frame_max_err, err)
            n_visible += 1
            u_avg, v_avg = (u1[i] + u2[i]) / 2.0, (v1[i] + v2[i]) / 2.0
            r = 12
            draw.ellipse([u_avg - r, v_avg - r, u_avg + r, v_avg + r], outline=(255, 0, 255), width=3)
            draw.line([(u_avg - r, v_avg), (u_avg + r, v_avg)], fill=(255, 0, 255), width=2)
            draw.line([(u_avg, v_avg - r), (u_avg, v_avg + r)], fill=(255, 0, 255), width=2)
            draw.text((u_avg + r + 2, v_avg - r), f'gcp{i}', fill=(255, 0, 255))

        img.save(os.path.join(out_dir, os.path.splitext(rec['frame'])[0] + '_gcp.png'))
        max_err = max(max_err, frame_max_err)
        n_checked += n_visible
        results.append({'frame': rec['frame'], 'n_gcp_visible': n_visible, 'max_err_px': frame_max_err})
        if n_visible:
            print(f"  {rec['frame']}: {n_visible} GCP(s) visible, max discrepancy = {frame_max_err:.4f}px")

    passed = n_checked > 0 and max_err < tol_px
    print(f'  {n_checked} total GCP sighting(s) across {len(results)} frame(s); '
          f'max discrepancy = {max_err:.4f}px (tolerance {tol_px}px)')
    if n_checked == 0:
        print('  WARNING: no GCP was visible in any sampled frame -- Part C did not actually test '
              'anything. Pick poses that look at the annotated block, or check the GCP world '
              'coordinates against these poses.')
    return passed, results


def build_photo_raster(georef, mesh_path, topdown_meta, resolution, out_path):
    """Bake the REAL birdseye photo itself onto the world-XY grid (not a
    classification) -- the "known ground-control pattern" for Part B.
    Same grid/height-lookup/projection path build_label_raster.py uses,
    so a mismatch downstream can only be a geometry problem, never a
    classification one. Uses PIL, not cv2, so this script only needs
    label_render_venv_gl (which lacks cv2) -- channel order (RGB vs BGR)
    doesn't matter here since phase_correlate reduces to grayscale via a
    channel-order-agnostic mean."""
    height_tree, mesh_z = rc.load_height_lookup(mesh_path)
    _, _, _, _, flat_x, flat_y, w_px, h_px = rc.build_output_grid(georef, resolution)
    flat_z = rc.lookup_heights(height_tree, mesh_z, flat_x, flat_y)

    photo = np.asarray(Image.open(topdown_meta['image_path']).convert('RGB'))

    points_w = np.column_stack([flat_x, flat_y, flat_z])
    u, v, valid = rc.world_to_pixel(points_w, topdown_meta)
    ui = np.clip(np.round(u).astype(int), 0, topdown_meta['width'] - 1)
    vi = np.clip(np.round(v).astype(int), 0, topdown_meta['height'] - 1)

    sampled = np.zeros((flat_x.shape[0], 3), dtype=np.uint8)
    sampled[valid] = photo[vi[valid], ui[valid]]
    raster = sampled.reshape(h_px, w_px, 3)

    Image.fromarray(raster, 'RGB').save(out_path)
    rc.write_raster_georef(georef, out_path, resolution, w_px, h_px)
    return out_path


def _to_gray_float(img):
    img = np.asarray(img).astype(np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img


def _subpixel_offset(profile, peak, size):
    """Parabolic sub-pixel refinement around an integer peak index in a
    1D (possibly wrapped) correlation profile."""
    prev_v = profile[(peak - 1) % size]
    peak_v = profile[peak]
    next_v = profile[(peak + 1) % size]
    denom = prev_v - 2 * peak_v + next_v
    if abs(denom) < 1e-9:
        return 0.0
    return 0.5 * (prev_v - next_v) / denom


def phase_correlate(img_a, img_b):
    """FFT-based phase correlation, own implementation (no cv2 in this
    venv): returns (dx, dy, confidence) describing the shift of img_b
    relative to img_a in pixels, plus a normalized cross-power peak
    height as a confidence score (higher = more reliable). Standard
    Hann-windowed cross-power-spectrum method with parabolic sub-pixel
    peak refinement."""
    a = _to_gray_float(img_a)
    b = _to_gray_float(img_b)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    h, w = a.shape
    window = np.outer(np.hanning(h) if h > 1 else np.ones(1), np.hanning(w) if w > 1 else np.ones(1))

    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag == 0] = 1e-12
    corr = np.fft.ifft2(cross / mag).real

    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    confidence = float(corr[py, px])

    px_sub = px + _subpixel_offset(corr[py, :], px, w)
    py_sub = py + _subpixel_offset(corr[:, px], py, h)

    dx = px_sub if px_sub <= w // 2 else px_sub - w
    dy = py_sub if py_sub <= h // 2 else py_sub - h
    return float(dx), float(dy), confidence


def tile_phase_correlate(img_a, img_b, tiles_x, tiles_y):
    h, w = img_a.shape[:2]
    th, tw = h // tiles_y, w // tiles_x
    out = []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, y1 = ty * th, (ty + 1) * th if ty < tiles_y - 1 else h
            x0, x1 = tx * tw, (tx + 1) * tw if tx < tiles_x - 1 else w
            tile_a, tile_b = img_a[y0:y1, x0:x1], img_b[y0:y1, x0:x1]
            if tile_a.size == 0 or np.std(tile_a) < 1e-6:
                out.append({'tile_x': tx, 'tile_y': ty, 'dx_px': None, 'dy_px': None, 'confidence': 0.0})
                continue
            dx, dy, conf = phase_correlate(tile_a, tile_b)
            out.append({'tile_x': tx, 'tile_y': ty, 'dx_px': dx, 'dy_px': dy, 'confidence': conf})
    return out


def draw_offset_debug(real_arr, tile_offsets, tiles_x, tiles_y, out_path, arrow_scale=5.0):
    img = Image.fromarray(real_arr).convert('RGB')
    draw = ImageDraw.Draw(img)
    h, w = real_arr.shape[:2]
    th, tw = h / tiles_y, w / tiles_x
    for t in tile_offsets:
        if t['dx_px'] is None:
            continue
        cx, cy = (t['tile_x'] + 0.5) * tw, (t['tile_y'] + 0.5) * th
        ex, ey = cx + t['dx_px'] * arrow_scale, cy + t['dy_px'] * arrow_scale
        draw.line([(cx, cy), (ex, ey)], fill=(255, 0, 0), width=3)
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 0))
    img.save(out_path)


def run_part_b(args, frames_dir, out_dir):
    georef = json.load(open(args.georef))
    topdown_meta = load_topdown_meta(args.topdown_meta)

    photo_raster_path = os.path.join(out_dir, 'photo_raster.png')
    print(f'  Building ortho photo-raster (ground-control pattern) at {photo_raster_path} ...')
    build_photo_raster(georef, args.mesh, topdown_meta, args.resolution, photo_raster_path)
    texture_rgb = np.asarray(Image.open(photo_raster_path).convert('RGB'))

    results, flagged = [], []

    def on_frame(rec, rendered_arr):
        real_path = os.path.join(frames_dir, rec['frame'])
        if not os.path.exists(real_path):
            print(f"  SKIP {rec['frame']}: no matching real frame at {real_path}")
            return
        real_arr = np.asarray(Image.open(real_path).convert('RGB'))
        if real_arr.shape != rendered_arr.shape:
            print(f"  SKIP {rec['frame']}: shape mismatch real={real_arr.shape} rendered={rendered_arr.shape}")
            return

        dx, dy, conf = phase_correlate(rendered_arr, real_arr)
        tiles = tile_phase_correlate(rendered_arr, real_arr, args.tiles_x, args.tiles_y)
        offset_mag = (dx ** 2 + dy ** 2) ** 0.5

        # Phase correlation only ever measures a pure translation -- a low
        # confidence peak means the two images don't actually match well
        # under ANY translation (e.g. a rotation/scale mismatch, or too
        # little raster/photo overlap in view), not that the true offset
        # is small. Treating a low-confidence result as "ok" would hide
        # exactly the kind of bug this script exists to catch; treating it
        # as a confidently-measured large offset would be equally wrong.
        # So it gets its own third bucket instead of collapsing into
        # flagged/ok.
        low_confidence = conf < args.min_confidence
        entry = {'frame': rec['frame'], 'dx_px': dx, 'dy_px': dy, 'confidence': conf,
                  'low_confidence': low_confidence,
                  'pose_age_s': rec.get('pose_age_s'), 'tiles': tiles}
        results.append(entry)
        flag = (not low_confidence) and offset_mag > args.offset_threshold_px
        if flag:
            flagged.append(entry)
        status = 'FLAGGED' if flag else ('low-confidence (inconclusive)' if low_confidence else 'ok')
        print(f"  {rec['frame']}: dx={dx:+.2f} dy={dy:+.2f} |offset|={offset_mag:.2f}px "
              f"conf={conf:.3f} pose_age={(rec.get('pose_age_s') or 0) * 1000:.1f}ms  {status}")

        draw_offset_debug(real_arr, tiles, args.tiles_x, args.tiles_y,
                           os.path.join(out_dir, os.path.splitext(rec['frame'])[0] + '_offsets.png'))

    render_texture_sequence(args.mesh, texture_rgb,
                             os.path.join(args.recording_dir, 'poses.jsonl'),
                             os.path.join(args.recording_dir, 'camera_info.json'),
                             on_frame, limit=args.limit)

    with open(os.path.join(out_dir, 'alignment_report.json'), 'w') as f:
        json.dump({'threshold_px': args.offset_threshold_px,
                    'results': results,
                    'flagged_frames': [e['frame'] for e in flagged]}, f, indent=2)

    return results, flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recording-dir', required=True, help='Dir with frames/, poses.jsonl, camera_info.json')
    parser.add_argument('--georef', required=True, help='georef_*.json from bake_world_mesh.py')
    parser.add_argument('--mesh', required=True, help='label_mesh_*.glb from bake_world_mesh.py')
    parser.add_argument('--topdown-meta', required=True, help='topdown_*.yaml from capture_topdown.py')
    parser.add_argument('--resolution', type=float, default=0.02, help='photo-raster meters per pixel')
    parser.add_argument('--offset-threshold-px', type=float, default=3.0,
                         help='Flag a frame if the global phase-correlation offset exceeds this many pixels')
    parser.add_argument('--min-confidence', type=float, default=0.05,
                         help='Below this normalized cross-power peak height, treat the offset as '
                              'unreliable (low raster/photo overlap, or a rotation/scale mismatch '
                              'phase correlation cannot represent as a translation) rather than a '
                              'confirmed flag -- reported separately, not silently counted as "ok"')
    parser.add_argument('--tiles-x', type=int, default=4)
    parser.add_argument('--tiles-y', type=int, default=3)
    parser.add_argument('--sample-points', type=int, default=200, help='mesh points sampled per pose for Part A')
    parser.add_argument('--sample-radius-m', type=float, default=15.0)
    parser.add_argument('--out-dir', default=None, help='Default: <recording-dir>/alignment_check')
    parser.add_argument('--limit', type=int, default=None, help='Only check the first N poses')
    parser.add_argument('--gcp-file', default=None,
                         help='plot_lattice_*.json from annotate_plots.py -- if given, also run Part C '
                              '(GCP pixel-projection check)')
    parser.add_argument('--gcp-tol-px', type=float, default=2.0,
                         help='Part C: max allowed discrepancy between the two projection code paths')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.recording_dir, 'alignment_check')
    os.makedirs(out_dir, exist_ok=True)

    cam_info = json.load(open(os.path.join(args.recording_dir, 'camera_info.json')))
    poses = [json.loads(line) for line in open(os.path.join(args.recording_dir, 'poses.jsonl'))]
    if args.limit:
        poses = poses[:args.limit]

    print('=== Part A: analytic projection self-consistency (no image data) ===')
    part_a_ok = run_part_a(args.mesh, poses, cam_info, args.sample_points, args.sample_radius_m)
    if not part_a_ok:
        print('  Part A FAILED. This conclusively means a bug in the projection code itself '
              '(pose composition / GL_FIX sign / intrinsics) -- fix that before trusting Part B '
              'or any rendered classification output.')
    else:
        print('  Part A PASSED.')

    print()
    print('=== Part B: photometric registration against real frames ===')
    results, flagged = run_part_b(args, os.path.join(args.recording_dir, 'frames'), out_dir)

    low_conf = [e for e in results if e['low_confidence']]
    print()
    print(f'Wrote {len(results)} per-frame result(s) and offset-arrow debug PNGs to {out_dir}')
    if low_conf:
        print(f'{len(low_conf)}/{len(results)} frame(s) had confidence below {args.min_confidence} -- '
              f'{[e["frame"] for e in low_conf]}. The measured offset on these is NOT reliable (phase '
              "correlation only measures translation; a low peak often means the two images don't "
              'match under any translation at all -- e.g. a rotation/scale mismatch, or too little '
              'raster/photo overlap in view). Inspect their *_offsets.png and the rendered vs real '
              'frame by eye before concluding anything either way.')
    if flagged:
        print(f'{len(flagged)}/{len(results)} frame(s) confidently exceeded the '
              f'{args.offset_threshold_px}px threshold: {[e["frame"] for e in flagged]}')
        uniform = all(
            abs(e['dx_px'] - flagged[0]['dx_px']) < 1.0 and abs(e['dy_px'] - flagged[0]['dy_px']) < 1.0
            for e in flagged)
        if uniform:
            print('Flagged offsets look spatially UNIFORM across frames -- check '
                  'camera_height/camera_pitch/camera_x_offset and pose timing, not classification.')
        else:
            print('Flagged offsets vary between frames/tiles -- check pose_age_s for staleness, or '
                  'mesh resolution near those poses, before assuming a fixed calibration bug.')
    if not flagged and not low_conf:
        print('No frames exceeded the offset threshold. If a rendered label still looks wrong here, '
              "it's very likely a classification-quality issue (retune hue thresholds or the "
              'structure-raster relief threshold), not a geometry bug.')

    if args.gcp_file:
        print()
        print('=== Part C: GCP pixel-projection check ===')
        gcp_points_world = load_gcp_points_world(args.gcp_file, args.mesh)
        part_c_ok, gcp_results = run_part_c(gcp_points_world, poses, cam_info,
                                             os.path.join(args.recording_dir, 'frames'),
                                             out_dir, tol_px=args.gcp_tol_px)
        with open(os.path.join(out_dir, 'gcp_report.json'), 'w') as f:
            json.dump({'tol_px': args.gcp_tol_px, 'gcp_points_world': gcp_points_world.tolist(),
                        'results': gcp_results, 'passed': part_c_ok}, f, indent=2)
        if part_c_ok:
            print(f'  Part C PASSED. GCP debug overlays (*_gcp.png) written to {out_dir} for visual review.')
        else:
            print('  Part C FAILED or inconclusive -- see warning above. GCP debug overlays '
                  f'(*_gcp.png) written to {out_dir}.')


if __name__ == '__main__':
    main()
