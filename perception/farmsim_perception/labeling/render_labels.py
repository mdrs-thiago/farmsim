"""
render_labels.py

Phase 5 of the mesh-based label-rendering pipeline: for every recorded
camera pose (Phase 4's poses.jsonl), render the label mesh (Phase 2's
baked, planar-UV world mesh) from that exact pose, textured with BOTH the
vegetation raster (build_label_raster.py) and the structure raster
(build_structure_raster.py) at once. Because the label mesh IS the real
terrain surface, this naturally accounts for actual terrain height and
occlusion at render time -- no flat-plane assumption anywhere in this
step, unlike every projection this pipeline replaces.

The mesh/pose/render core lives in render_texture_sequence(), factored
out of main() so other callers can reuse the exact same, already-correct
render path against a DIFFERENT texture: check_alignment.py's photometric
diagnostic renders the real birdseye photo itself (not a class raster)
through this same function, to tell a genuine reprojection bug apart from
a bad classification.

main() below packs both class-id rasters into one texture (vegetation in
the R byte, structure in the G byte) and renders them in a SINGLE GL
pass, then splits the result back into two separately-named PNG streams.
This is safe only because NEAREST texture filtering is already mandatory
here to avoid class-boundary bleed -- under LINEAR filtering, packing two
independent channels into one texture would silently blend R/G bytes
across a class boundary and corrupt both channels at once. Single-pass
also guarantees the two channels are pixel-exact co-registered by
construction, which the eventual downstream fusion of these two channels
(explicitly out of scope here) will depend on. The packing is a
render-time-only optimization -- every file written to disk stays
single-channel/single-purpose, same as if two full passes had been run.

Implementation note: originally written against pyrender/PyOpenGL, per
the plan this pipeline follows. PyOpenGL 3.1.0 (its only release in
years) turned out to be incompatible with this machine's Python 3.14 --
its ctypes array-marshalling for glGenTextures raises
"No array-type handler for type _ctypes.CArgObject" unconditionally, a
library/interpreter incompatibility with no workaround found (confirmed:
fails identically calling the wrapped function, the raw base function,
and with an explicit output array). Rewritten on moderngl instead --
actively maintained, and this machine turned out to have a real GPU
(NVIDIA RTX 3050) that PyOpenGL's EGL path never surfaced either.
Same three failure modes still apply and are handled explicitly below:
unlit shading, the ROS-optical -> OpenGL axis convention, and
camera-to-world vs world-to-camera pose direction.

Usage:
  python -m farmsim_perception.labeling.render_labels \
      --mesh farm/fields/20250425/label_mesh.glb \
      --vegetation-raster farm/fields/20250425/vegetation_raster.png \
      --structure-raster farm/fields/20250425/structure_raster.png \
      --poses recordings/pose_test/poses.jsonl \
      --camera-info recordings/pose_test/camera_info.json \
      --out-dir recordings/pose_test/label_frames \
      --limit 5
"""

import argparse
import json
import os

import moderngl
import numpy as np
import trimesh
from PIL import Image

from farmsim_farm.palettes import STRUCT_PALETTE, VEG_PALETTE, classid_to_rgb, rgb_to_classid

# OpenGL camera convention (-Z forward, +Y up) vs the T_world_optical
# convention already used throughout this codebase (+Z forward, +Y down,
# +X right -- see live_label_teleop.py's sensor->optical conversion).
GL_FIX = np.diag([1.0, -1.0, -1.0, 1.0])

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_uv = in_uv;
}
"""

# Unlit: sample the label texture directly, no lighting term at all -- a
# shaded/sunlit difference in the (semantically meaningless) label colors
# must never change what class a pixel reads as.
FRAGMENT_SHADER = """
#version 330
uniform sampler2D label_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(label_tex, v_uv);
}
"""


def intrinsics_to_projection(fx, fy, cx, cy, width, height, near, far):
    """Standard pinhole-intrinsics -> OpenGL clip-space projection matrix
    (column convention: clip = P @ view_space_point, +X right/+Y up/-Z
    forward in view space already, i.e. applied AFTER GL_FIX)."""
    return np.array([
        [2 * fx / width, 0, 1 - 2 * cx / width, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ])


def render_texture_sequence(mesh_path, texture_rgb, poses_path, camera_info_path, on_frame,
                             limit=None, near=0.1, far=100.0):
    """Core render loop: load the baked label mesh once, upload
    `texture_rgb` (an already-built (H,W,C) uint8 array -- the caller
    decides what it encodes: a recolored class raster, a packed
    multi-channel class-id array, or a real photo) as a NEAREST-filtered
    texture (no interpolation across class boundaries), then render it
    from every pose in `poses_path` against `camera_info_path`'s
    intrinsics. Calls `on_frame(rec, arr)` once per rendered frame with
    the pose record (the raw dict from poses.jsonl) and the rendered
    (H,W,C) uint8 array (row 0 = top, already flipped back from OpenGL's
    bottom-row-0 convention) -- the caller decides what to do with it.
    """
    with open(camera_info_path) as f:
        cam_info = json.load(f)
    fx, fy, cx, cy = cam_info['fx'], cam_info['fy'], cam_info['cx'], cam_info['cy']
    width, height = cam_info['width'], cam_info['height']

    print(f'Loading label mesh {mesh_path} ...')
    m = trimesh.load(mesh_path, process=False)
    if isinstance(m, trimesh.Scene):
        m = list(m.geometry.values())[0]
    uv = m.visual.uv
    if uv is None:
        raise RuntimeError(f'{mesh_path} has no UVs -- run bake_world_mesh.py first')
    vertices = m.vertices.astype('f4')
    uv = uv.astype('f4')
    # OpenGL texture sampling maps v=0 to the FIRST row of the uploaded
    # data (texel row 0), full stop -- that's just array indexing, not a
    # convention that can "match" a different row convention on its own.
    # Phase 2's UV (v = (y - y0) / H, world-y-up) has v=0 at world's MIN y,
    # but Phase 3's raster is written with row 0 = MAX world y
    # (raster_common.build_output_grid, image-style top-left origin) --
    # i.e. row 0 of the uploaded texture data is world-y-MAX, which v=0
    # would otherwise incorrectly sample as if it were world-y-MIN.
    # Empirically confirmed (not just derived): a world point near the
    # raster's "row 0 = max-y" half rendered from the WRONG half of a
    # red/blue-split test texture without this flip, and the correct half
    # with it. Flip v here so v=0 (world-y MIN) samples the raster's last
    # row (also world-y MIN), matching.
    uv[:, 1] = 1.0 - uv[:, 1]
    indices = m.faces.astype('i4')

    n_components = texture_rgb.shape[2]

    ctx = moderngl.create_context(standalone=True, backend='egl')
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    vbo_pos = ctx.buffer(vertices.tobytes())
    vbo_uv = ctx.buffer(uv.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo_pos, '3f', 'in_position'), (vbo_uv, '2f', 'in_uv')],
        ibo,
    )

    tex = ctx.texture((texture_rgb.shape[1], texture_rgb.shape[0]), n_components,
                       np.ascontiguousarray(texture_rgb).tobytes())
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)  # no interpolation across class boundaries
    tex.use(location=0)
    prog['label_tex'].value = 0

    color_rbo = ctx.renderbuffer((width, height), components=n_components)
    depth_rbo = ctx.depth_renderbuffer((width, height))
    fbo = ctx.framebuffer(color_attachments=[color_rbo], depth_attachment=depth_rbo)
    fbo.use()
    ctx.enable(moderngl.DEPTH_TEST)

    projection = intrinsics_to_projection(fx, fy, cx, cy, width, height, near, far)

    poses = []
    with open(poses_path) as f:
        for line in f:
            poses.append(json.loads(line))
    if limit:
        poses = poses[:limit]
    print(f'Rendering {len(poses)} frame(s)...')

    for rec in poses:
        cam_to_world = np.array(rec['T_world_optical']) @ GL_FIX
        view = np.linalg.inv(cam_to_world)
        mvp = projection @ view
        prog['mvp'].write(mvp.T.astype('f4').tobytes())  # moderngl expects column-major

        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        vao.render(moderngl.TRIANGLES)

        raw = fbo.read(components=n_components, alignment=1)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, n_components)
        arr = arr[::-1, :, :]  # FBO row 0 is the image's bottom row; flip to top-down

        on_frame(rec, arr)


def _load_classid_raster(path, palette):
    """Load a raster PNG written by build_label_raster.py/
    build_structure_raster.py and recover its class-id array. Those
    scripts save `classid_to_rgb(...)[:, :, ::-1]` (their palette tuples
    are BGR-labeled, reversed once to get true RGB bytes on disk) -- so
    loading needs the same reversal applied again before rgb_to_classid's
    exact-tuple-match lookup against the (still BGR-labeled) palette."""
    file_rgb = np.asarray(Image.open(path).convert('RGB'))
    return rgb_to_classid(file_rgb[:, :, ::-1], palette)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--vegetation-raster', required=True, help='vegetation_raster_*.png from build_label_raster.py')
    parser.add_argument('--structure-raster', required=True, help='structure_raster_*.png from build_structure_raster.py')
    parser.add_argument('--poses', required=True, help='poses.jsonl from record_camera_poses.py')
    parser.add_argument('--camera-info', required=True, help='camera_info.json from record_camera_poses.py')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--limit', type=int, default=None, help='Render only the first N poses')
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100.0)
    args = parser.parse_args()

    veg_dir = os.path.join(args.out_dir, 'vegetation')
    struct_dir = os.path.join(args.out_dir, 'structure')
    os.makedirs(veg_dir, exist_ok=True)
    os.makedirs(struct_dir, exist_ok=True)

    print(f'Loading vegetation raster {args.vegetation_raster} ...')
    veg_ids = _load_classid_raster(args.vegetation_raster, VEG_PALETTE)
    print(f'Loading structure raster {args.structure_raster} ...')
    struct_ids = _load_classid_raster(args.structure_raster, STRUCT_PALETTE)

    if veg_ids.shape != struct_ids.shape:
        raise ValueError(
            f'vegetation raster {veg_ids.shape} and structure raster {struct_ids.shape} must be the '
            'same pixel size -- build both from the same --georef/--resolution')

    # See module docstring: safe only because NEAREST filtering is
    # already mandatory. B channel is unused/reserved.
    packed = np.zeros((*veg_ids.shape, 3), dtype=np.uint8)
    packed[..., 0] = veg_ids
    packed[..., 1] = struct_ids

    def on_frame(rec, arr):
        base = os.path.splitext(rec['frame'])[0]
        veg_class, struct_class = arr[..., 0], arr[..., 1]

        veg_png = classid_to_rgb(veg_class, VEG_PALETTE)[:, :, ::-1]
        struct_png = classid_to_rgb(struct_class, STRUCT_PALETTE)[:, :, ::-1]
        Image.fromarray(veg_png, 'RGB').save(os.path.join(veg_dir, base + '_veg.png'))
        Image.fromarray(struct_png, 'RGB').save(os.path.join(struct_dir, base + '_struct.png'))

        covered = float((veg_class.astype(np.int32) + struct_class.astype(np.int32) > 0).mean())
        print(f"  {rec['frame']}: {base}_veg.png / {base}_struct.png  mesh coverage={covered:.2%}")

    render_texture_sequence(args.mesh, packed, args.poses, args.camera_info, on_frame,
                             limit=args.limit, near=args.near, far=args.far)
    print(f'Done. Wrote vegetation frames to {veg_dir} and structure frames to {struct_dir}')


if __name__ == '__main__':
    main()
