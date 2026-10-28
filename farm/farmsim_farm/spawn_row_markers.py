"""
spawn_row_markers.py

Drops small, brightly-colored sphere models into a running gz-sim world at
the world (x, y) points from a rows.yaml (segment_rows.py / mark_rows.py
output) -- one color per row id. Two purposes:

  1. Visual sanity check: open gz-sim with a GUI and you can *see* the
     markers sitting on the actual terrain, right on top of (or off of)
     the real soil path -- a much more direct check than eyeballing a
     static 2D overlay image.
  2. A physical, known-world-coordinate reference for later automatic
     label generation: whatever records the front camera later knows its
     own pose (from /odom, which is exact sim ground truth -- see
     row_follower.py), so any marker's pixel position in a given frame is
     a pure projection computation, no re-detection needed. They also
     happen to be visually distinct (bright, saturated, matte colors
     against green crop/tan soil) if a vision-based check against the
     recorded video is ever wanted instead.

Requires a world already running (launch farm_sim.launch.py or
capture_topdown.launch.py first) -- this calls `ros_gz_sim create` once
per marker against that running world, the same mechanism
farm_sim.launch.py uses to spawn the robot itself.

IMPORTANT: export the same GZ_PARTITION the world's launch file set
(capture_topdown.launch.py: thorvald_topdown_capture; farm_sim.launch.py:
thorvald_farm_sim) in the shell running this script first. Without it,
`ros_gz_sim create` talks to gz-transport's default partition, silently
finds no matching world/service, and hangs indefinitely instead of
erroring -- confirmed empirically (the first attempt sat for 5+ minutes
on marker 1 of 37 before being killed).

Default marker --z (0.6) is above this survey's max mesh elevation
(0.421, from the vertex range noted in vollebekk_20250624.sdf) --
confirmed empirically that markers placed near/below the terrain
surface get rendered *under* the crop canopy mesh and are invisible from
directly above, even though the spawn itself reports success.

Usage (row markers, along a marked corridor):
  python3 spawn_row_markers.py --rows-file config/rows_20250624.yaml \
      --world vollebekk_farm_20250624

Usage (GCP markers, one per annotated plot-block corner -- for Step 4
verification: project each corner's known world XYZ into a rendered RGB
frame and its matching mask, and confirm they land on identical pixels):
  python3 spawn_row_markers.py --gcp-file config/plot_lattice_20250425.json \
      --world vollebekk_farm_20250425
"""

import argparse
import json
import subprocess

import numpy as np
import yaml

# Matches draw_overlay's per-row color convention in segment_rows.py
# (BGR there; these are RGB 0-1 for SDF <material>).
ROW_COLORS = [
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.65, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.5),
    (0.5, 0.0, 1.0),
]

# GCP markers use a fixed, visually-distinct-from-ROW_COLORS palette (pure
# white first -- classic photogrammetric-GCP convention -- then a small
# fixed set of saturated colors) since there are always exactly as many as
# annotate_plots.py's 4 clicked block corners, never per-row.
GCP_COLORS = [
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
]


def resample_polyline(points, spacing):
    pts = np.array(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total < 1e-6:
        return [tuple(pts[0])]
    n = max(2, int(total // spacing) + 1)
    targets = np.linspace(0.0, total, n)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = []
    for t in targets:
        idx = min(max(np.searchsorted(cum, t) - 1, 0), len(seg_len) - 1)
        local_t = (t - cum[idx]) / seg_len[idx] if seg_len[idx] > 1e-9 else 0.0
        p = pts[idx] + local_t * seg[idx]
        out.append((float(p[0]), float(p[1])))
    return out


def marker_sdf(name, radius, color):
    r, g, b = color
    return (
        '<sdf version="1.8">'
        f'<model name="{name}">'
        '<static>true</static>'
        '<link name="link">'
        '<visual name="visual">'
        f'<geometry><sphere><radius>{radius}</radius></sphere></geometry>'
        '<material>'
        f'<ambient>{r} {g} {b} 1</ambient>'
        f'<diffuse>{r} {g} {b} 1</diffuse>'
        f'<emissive>{r * 0.6} {g * 0.6} {b * 0.6} 1</emissive>'
        '</material>'
        '</visual>'
        '</link>'
        '</model>'
        '</sdf>'
    )


def _spawn_one(name, radius, color, x, y, z, world):
    cmd = [
        'ros2', 'run', 'ros_gz_sim', 'create',
        '-world', world,
        '-string', marker_sdf(name, radius, color),
        '-name', name,
        '-allow_renaming', 'true',
        '-x', str(x), '-y', str(y), '-z', str(z),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Failed to spawn {name}: {result.stderr.strip()}')
        return False
    return True


def spawn_row_markers(rows_file, world, spacing, radius, z):
    with open(rows_file, 'r') as f:
        data = yaml.safe_load(f)

    total = 0
    for row in data['rows']:
        rid = row['id']
        color = ROW_COLORS[rid % len(ROW_COLORS)]
        for i, (x, y) in enumerate(resample_polyline(row['points'], spacing)):
            if _spawn_one(f'row{rid}_marker{i}', radius, color, x, y, z, world):
                total += 1
    print(f'Spawned {total} row marker(s) from {rows_file}')


def spawn_gcp_markers(gcp_file, world, radius, z):
    """One marker per plot_lattice_*.json corner (annotate_plots.py's
    'corners_world_xy', in click order) -- a small, fixed set (always 4),
    unlike the row markers' per-row resampling."""
    with open(gcp_file, 'r') as f:
        lattice = json.load(f)

    total = 0
    for i, (x, y) in enumerate(lattice['corners_world_xy']):
        color = GCP_COLORS[i % len(GCP_COLORS)]
        if _spawn_one(f'gcp{i}', radius, color, x, y, z, world):
            total += 1
    print(f'Spawned {total} GCP marker(s) from {gcp_file}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rows-file', default=None, help='Path to a rows.yaml')
    group.add_argument('--gcp-file', default=None,
                        help='Path to a plot_lattice_*.json from annotate_plots.py -- spawns one '
                             "marker per annotated block corner instead of per row")
    parser.add_argument('--world', default='vollebekk_farm_20250624',
                        help='gz-sim world name (the <world name="..."> in the SDF)')
    parser.add_argument('--spacing', type=float, default=8.0,
                        help='(--rows-file only) distance in meters between markers along each row')
    parser.add_argument('--radius', type=float, default=0.25, help='Marker sphere radius, meters')
    parser.add_argument('--z', type=float, default=0.6,
                        help='Marker world z -- must clear the crop canopy mesh (this '
                             "survey's max elevation is 0.421) or markers render hidden "
                             'underneath it, invisible from directly above')
    args = parser.parse_args()

    if args.gcp_file:
        spawn_gcp_markers(args.gcp_file, args.world, args.radius, args.z)
    else:
        spawn_row_markers(args.rows_file, args.world, args.spacing, args.radius, args.z)


if __name__ == '__main__':
    main()
