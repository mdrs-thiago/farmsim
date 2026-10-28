"""
bake_world_mesh.py

Phases 1-2 of the mesh-based label-rendering pipeline: load the raw
photogrammetry terrain mesh, compose the SDF's model-level <pose> into a
4x4 transform, bake it into a WORLD-COORDINATE copy of the mesh (so
everything downstream -- UV assignment, the label raster, the renderer --
works directly in world XY without re-deriving the transform each time),
and assign fresh planar UVs from the baked mesh's own world XY bounding
box.

Why bake instead of carrying the transform through the renderer: fewer
places to get the composition order or convention wrong. One transform,
applied once, checked once (see the bounds sanity print below).

Why NOT trust the raw mesh's own UVs or its raw vertex bounds directly:
- This photogrammetry export's raw .obj has ~2000 vertices unreferenced
  by any face (confirmed for the 20250425 survey: trimesh's mesh-aware
  bounds vs a naive raw "v x y z" line parse differ by >20m on Z alone) --
  almost certainly stray reconstruction artifacts (sky/drone reflections).
  Loading through trimesh (not hand-parsing) avoids being fooled by them.
- The existing texture UVs are whatever ODM's photogrammetry pipeline
  produced (12+ non-rectilinear per-material atlases per prior
  investigation in this codebase) -- not a usable world-XY mapping.
  A label mesh needs its own planar UVs assigned by construction instead.

Usage:
  python farm/tools/bake_world_mesh.py \
      --mesh /path/to/odm_textured_model_geo.obj \
      --pose 33.8 -13.8 -116.9 0 0 1.066 \
      --out-mesh farm/fields/20250425/label_mesh.glb \
      --out-georef farm/fields/20250425/georef.json
"""

import argparse
import json

import numpy as np
import trimesh


def sdf_pose_to_matrix(x, y, z, roll, pitch, yaw):
    """SDF pose convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    r = rz @ ry @ rx
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = [x, y, z]
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True, help='Path to the raw .obj terrain mesh')
    parser.add_argument('--pose', type=float, nargs=6, required=True,
                         metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                         help="The terrain model's SDF <pose>, e.g. from "
                              'worlds/vollebekk_<date>.sdf\'s <include><pose>')
    parser.add_argument('--out-mesh', required=True,
                         help='Path to write the baked, world-coordinate label mesh (.glb)')
    parser.add_argument('--out-georef', required=True,
                         help='Path to write the georeference JSON (x0,y0,W,H + the '
                              'composed transform, for anything downstream that still '
                              'needs it)')
    args = parser.parse_args()

    print(f'Loading {args.mesh} ...')
    m = trimesh.load(args.mesh, process=False, force='mesh')
    print(f'  {len(m.vertices)} vertices, {len(m.faces)} faces')
    print(f'  mesh-local bounds:\n{m.bounds}')

    transform = sdf_pose_to_matrix(*args.pose)
    print(f'Composed mesh-local -> world transform (pose {args.pose}):\n{transform}')

    m.apply_transform(transform)
    print(f'  world bounds:\n{m.bounds}')

    x0, y0 = m.bounds[0, 0], m.bounds[0, 1]
    x1, y1 = m.bounds[1, 0], m.bounds[1, 1]
    w, h = x1 - x0, y1 - y0
    print(f'World XY footprint: x0={x0:.3f} y0={y0:.3f} W={w:.3f} H={h:.3f}')

    # Caveat from the plan this pipeline follows: this only works cleanly
    # because farm terrain is a heightfield (one z per xy) -- a face with a
    # near-vertical normal (overhang/wall) gets a degenerate (near-zero
    # area) planar UV. Fine for farm terrain; flag it rather than fail
    # silently.
    normals = m.face_normals
    vertical_frac = float((np.abs(normals[:, 2]) < 0.05).mean())
    if vertical_frac > 0.01:
        print(f'WARNING: {vertical_frac:.1%} of faces are near-vertical '
              '(overhang/wall-like) -- planar UV will be degenerate there. '
              'Fine for farm terrain if this stays small; investigate if not.')

    uv = np.column_stack([
        (m.vertices[:, 0] - x0) / w,
        (m.vertices[:, 1] - y0) / h,
    ])
    m.visual = trimesh.visual.TextureVisuals(uv=uv)

    m.export(args.out_mesh)
    print(f'Wrote baked, UV-mapped world mesh to {args.out_mesh}')

    georef = {
        'x0': float(x0), 'y0': float(y0), 'width_m': float(w), 'height_m': float(h),
        'z_min': float(m.bounds[0, 2]), 'z_max': float(m.bounds[1, 2]),
        'source_mesh': args.mesh,
        'sdf_pose': list(args.pose),
        'composed_transform': transform.tolist(),
    }
    with open(args.out_georef, 'w') as f:
        json.dump(georef, f, indent=2)
    print(f'Wrote georeference to {args.out_georef}')


if __name__ == '__main__':
    main()
