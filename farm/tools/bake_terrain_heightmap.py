#!/usr/bin/env python3
"""
bake_terrain_heightmap.py

Bakes a regular-grid terrain heightmap from the world-baked label mesh, so
weed/clutter placement can be STATIC instead of dropped-and-settled.

Why this exists: add_weeds originally spawned every weed as a light dynamic
body from drop_z and let gz-sim's physics settle it onto the terrain, because
terrain height at an arbitrary (x, y) wasn't known without a mesh dependency.
That works, but each weed then resolves contacts against a 375k-face
photogrammetry mesh on every physics step for the whole run. Measured cost:
raising the weed count from 5 to 18 drops the simulation's real-time factor
from 0.75 to 0.52 -- a 32% slowdown, which is a third less recorded data per
session. Field of view, by contrast, costs nothing (0.78 at hfov 1.75) and
sensor noise only ~7%.

With a baked heightmap, a weed can be emitted as <static>true</static> at the
right z directly: no contacts, no settling wait, no physics cost at all, so
clutter density stops trading against dataset size.

The grid is sampled by ray-casting straight down onto the mesh, which is the
same geometry the labels themselves are baked from -- so a weed sits on the
same surface the ground truth describes. Cells with no hit (outside the
surveyed area) are filled with NaN and callers fall back to dropping.

Run once per survey; the .npz is a build artifact, not something a session
regenerates.

Usage:
  python farm/tools/bake_terrain_heightmap.py \
      --mesh farm/fields/20250425/label_mesh.glb \
      --out farm/fields/20250425/terrain_heightmap.npz --resolution 0.25
"""

import argparse

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--resolution', type=float, default=0.25,
                    help='grid spacing in metres (default 0.25)')
    args = ap.parse_args()

    scene = trimesh.load(args.mesh)
    mesh = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene
    x0, y0, z0 = mesh.bounds[0]
    x1, y1, z1 = mesh.bounds[1]
    print(f'mesh bounds x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}] z[{z0:.2f},{z1:.2f}]')

    xs = np.arange(x0, x1 + args.resolution, args.resolution)
    ys = np.arange(y0, y1 + args.resolution, args.resolution)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    origins = np.column_stack([gx.ravel(), gy.ravel(),
                               np.full(gx.size, z1 + 10.0)])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (len(origins), 1))

    print(f'casting {len(origins):,} rays onto {len(mesh.faces):,} faces '
          f'({len(xs)} x {len(ys)} grid at {args.resolution} m)...')

    # Cast in chunks. A single 107k-ray batch was observed to come back with
    # every hit z offset by exactly the ray-origin height, while the identical
    # call on a few hundred rays returned correct world-space points -- a
    # large-batch quirk in trimesh's ray backend, not in this grid. Chunking
    # avoids it; the assertion below is the real guard, so a silent offset can
    # never reach the weed placement it feeds.
    heights = np.full(gx.size, np.nan, dtype=np.float32)
    chunk = 8192
    for start in range(0, len(origins), chunk):
        o = origins[start:start + chunk]
        d = directions[start:start + chunk]
        locations, index_ray, _ = mesh.ray.intersects_location(
            o, d, multiple_hits=False)
        if len(locations) == 0:
            continue
        # Highest hit per ray wins: the mesh is locally non-manifold in places
        # and a weed rests on the top surface.
        order = np.argsort(locations[:, 2])
        heights[start + index_ray[order]] = locations[order, 2]
        if start % (chunk * 8) == 0:
            print(f'  {start + len(o):,}/{len(origins):,}', end='\r', flush=True)
    print()

    finite = heights[np.isfinite(heights)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if not (z0 - 0.5 <= lo and hi <= z1 + 0.5):
            raise SystemExit(
                f'ABORT: sampled heights [{lo:.2f}, {hi:.2f}] fall outside the mesh z '
                f'bounds [{z0:.2f}, {z1:.2f}]. The ray backend returned something that '
                f'is not a world-space point; refusing to write a heightmap that would '
                f'bury or float every weed.')
    heights = heights.reshape(gx.shape)

    covered = np.isfinite(heights)
    print(f'  hit {covered.sum():,}/{heights.size:,} cells ({100*covered.mean():.1f}%)')
    print(f'  z range {np.nanmin(heights):.2f} .. {np.nanmax(heights):.2f} m')

    np.savez_compressed(args.out, heights=heights, x0=x0, y0=y0,
                        resolution=args.resolution,
                        nx=len(xs), ny=len(ys))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
