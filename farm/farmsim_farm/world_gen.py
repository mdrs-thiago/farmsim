"""
generate_run_params.py

Produces one randomized recording-session configuration for
record_random_session.sh: a lighting-varied world SDF (same terrain,
different sun direction/color + ambient/background), a spawn pose along a
randomly chosen row (with lateral/heading jitter so row_follower has real
correcting to do, not just "start dead-center"), and a randomized
row_follower perturb_std. Deterministic given the same --seed, so a
specific run can be reproduced.

record_dataset.py calls `generate(args)` directly and gets the run
parameters back as a dict. Run as a script it prints `KEY="VALUE"` lines
(eval-able from bash) and optionally writes the same dict as JSON.

Also scatters a randomized number of weed/clutter plant instances into
the generated world's row corridors (add_weeds) -- pure visual
distractor content, not reflected in the label pipeline's ground truth
(which is baked from the static pre-survey classification and has no
concept of runtime world clutter, by design -- see add_weeds' own
docstring for the reasoning).

Usage:
  python3 generate_run_params.py \
      --rows-file .../rows_20250425.yaml \
      --world-template .../vollebekk_20250425.sdf \
      --out-world-dir .../generated_worlds \
      --session-name run_20260101_120000_seed42 \
      --seed 42
"""

import argparse
import json
import math
import os
import random
import xml.etree.ElementTree as ET

import yaml


def randomize_world(template_path, out_path, rng):
    """Copy `template_path`'s world SDF with the sun light and scene
    ambient/background randomized -- the terrain <include>/<model> block
    is never touched by this function (nothing here looks at or edits
    it), so the baked mesh/rasters this survey already has stay valid;
    only the RENDERED RGB video's shading changes. Comments are dropped
    by ElementTree's parser -- harmless for a generated, throwaway
    variant. Returns the sun direction vector actually used (for
    logging)."""
    tree = ET.parse(template_path)
    root = tree.getroot()

    elevation_deg = rng.uniform(8, 85)
    azimuth_deg = rng.uniform(0, 360)
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    direction = (-math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a), -math.sin(e))

    def jitter_color(base, lo=0.45, hi=1.30):
        return tuple(min(1.0, max(0.0, c * rng.uniform(lo, hi))) for c in base)

    sun = root.find(".//light[@name='sun']")
    sun.find('direction').text = f'{direction[0]:.4f} {direction[1]:.4f} {direction[2]:.4f}'
    diffuse = jitter_color((0.8, 0.8, 0.8))
    specular = jitter_color((0.2, 0.2, 0.2))
    sun.find('diffuse').text = f'{diffuse[0]:.3f} {diffuse[1]:.3f} {diffuse[2]:.3f} 1'
    sun.find('specular').text = f'{specular[0]:.3f} {specular[1]:.3f} {specular[2]:.3f} 1'

    scene = root.find('.//scene')
    ambient_base = tuple(float(v) for v in scene.find('ambient').text.split()[:3])
    background_base = tuple(float(v) for v in scene.find('background').text.split()[:3])
    ambient = jitter_color(ambient_base)
    background = jitter_color(background_base)
    scene.find('ambient').text = f'{ambient[0]:.3f} {ambient[1]:.3f} {ambient[2]:.3f} 1'
    scene.find('background').text = f'{background[0]:.3f} {background[1]:.3f} {background[2]:.3f} 1'

    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    return direction


WEED_MESH_URI = 'model://plants_disorganized/meshes/acanthus.stl'
# plants_disorganized's own Ogre material scripts (vrc/grass, vrc/cotton)
# don't render correctly under gz-sim's Ogre2 renderer -- confirmed
# visually (renders near-black/unlit instead of the intended green
# texture). Explicit ambient/diffuse/specular color is the reliable,
# renderer-native way to color an SDF <material> in gz-sim, used here
# instead of fighting the legacy Ogre script path.
WEED_BASE_SCALE = 0.005  # matches plants_disorganized/model.sdf's own per-plant scale
WEED_BASE_SCALE = 0.005  # matches plants_disorganized/model.sdf's own per-plant scale


def load_heightmap(path):
    """Terrain heightmap baked by bake_terrain_heightmap.py, or None if absent.

    Returns a sampler f(x, y) -> z (NaN outside the surveyed area), so weeds can
    be emitted STATIC at the correct ground height instead of being dropped and
    settled by physics. numpy only -- no mesh library needed at session-generation
    time, which is why this is a baked artifact rather than a live raycast."""
    if not path or not os.path.exists(path):
        return None
    import numpy as np
    d = np.load(path)
    heights = d['heights']
    x0, y0, res = float(d['x0']), float(d['y0']), float(d['resolution'])
    nx, ny = heights.shape

    def sample(x, y):
        i = int(round((x - x0) / res))
        j = int(round((y - y0) / res))
        if 0 <= i < nx and 0 <= j < ny:
            return float(heights[i, j])
        return float('nan')
    return sample


def spawn_height(heightmap, x, y, clearance=0.1, radius=1.3, fallback=2.0):
    """Spawn z just above the highest ground within `radius` of (x, y), so the
    robot settles onto the terrain instead of being dropped onto it. A long
    drop onto the photogrammetry mesh bounces it and it lands yawed and shifted."""
    if heightmap is None:
        return fallback
    import numpy as np
    zs = [heightmap(x + dx, y + dy) for dx in np.arange(-radius, radius + 1e-6, 0.25)
          for dy in np.arange(-radius, radius + 1e-6, 0.25) if dx * dx + dy * dy <= radius * radius]
    zs = [z for z in zs if not np.isnan(z)]
    return max(zs) + clearance if zs else fallback


def add_weeds(world_elem, rows_file, rng, n_weeds_range=(0, 10),
               corridor_offset_range=(0.4, 0.9), weed_scale_range=(0.6, 1.4), drop_z=1.5,
               heightmap=None):
    """Scatter a randomized number of weed/clutter plant instances into
    the world's row corridors, appended as new <model> children of
    `world_elem`. Pure visual distractor content: the label pipeline's
    ground truth (masks/bbox/lateral_offset/heading_error) is baked from
    the static pre-survey classification and has no concept of runtime
    world clutter -- by design, confirmed against the project's own
    intent, not an oversight. This trains/tests robustness to real-world
    clutter (a stray weed in the passage) against the TRUE row geometry,
    rather than trying to hand-label weed extent.

    Reuses plants_disorganized/meshes/acanthus.stl (a real plant mesh
    already on GZ_SIM_RESOURCE_PATH via ros_src) as a per-instance
    building block -- NOT plants_disorganized's own model.sdf, which is
    one fixed static layout of ~35 plants at fixed relative positions,
    not something placeable at an arbitrary world pose per instance.

    Terrain height at an arbitrary (x, y) isn't known in advance (the
    baked terrain mesh has real relief, and its own model pose carries a
    large z offset) -- rather than add a mesh-height-lookup dependency
    to this otherwise stdlib-only script, each weed is a light DYNAMIC
    body (not static) with a simple approximate collision cylinder,
    dropped from `drop_z` and left to settle onto the actual terrain via
    gz-sim's own physics. Same "don't know precise terrain height in
    advance, let gravity find it" approach farm_sim.launch.py's own
    `z` launch arg already documents and relies on for spawning the
    robot itself -- reused here, not reinvented. STARTUP_WAIT_S (12s,
    record_random_session.sh) already provides ample settle time for
    small light objects dropped from 1.5m.

    Each weed is placed near a randomly chosen row's centerline, offset
    laterally by corridor_offset_range (into the drivable corridor
    between beds, not on top of the bed itself) and at a random point
    along the row's length. Returns the number of weeds actually added.
    """
    with open(rows_file) as f:
        rows = yaml.safe_load(f)['rows']

    n_weeds = rng.randint(*n_weeds_range)
    for i in range(n_weeds):
        row = rng.choice(rows)
        pts = row['points']
        idx = rng.randrange(0, len(pts) - 1)
        x0, y0 = pts[idx]
        x1, y1 = pts[idx + 1]
        t = rng.uniform(0.0, 1.0)
        px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        heading = math.atan2(y1 - y0, x1 - x0)
        perp = heading + math.pi / 2
        side = rng.choice([-1.0, 1.0])
        offset = side * rng.uniform(*corridor_offset_range)
        wx, wy = px + offset * math.cos(perp), py + offset * math.sin(perp)
        yaw = rng.uniform(0.0, 2 * math.pi)
        scale = rng.uniform(*weed_scale_range) * WEED_BASE_SCALE

        # Static placement when the baked heightmap covers this point; falls
        # back to the original drop-and-settle otherwise (edges of the survey).
        # This is a large simulation-speed win, not a cosmetic change: dynamic
        # weeds resolve contacts against a 375k-face terrain mesh every physics
        # step, and measured real-time factor fell from 0.75 (5 weeds) to 0.52
        # (18 weeds) because of it -- a third less recorded data per session.
        # Static weeds cost nothing, so clutter density no longer trades
        # against dataset size.
        ground_z = heightmap(wx, wy) if heightmap is not None else float('nan')
        static = ground_z == ground_z  # False for NaN
        model = ET.SubElement(world_elem, 'model', {'name': f'weed_{i}'})
        ET.SubElement(model, 'static').text = 'true' if static else 'false'
        pose = ET.SubElement(model, 'pose')
        z = f'{ground_z:.3f}' if static else f'{drop_z}'
        pose.text = f'{wx:.3f} {wy:.3f} {z} 0 0 {yaw:.4f}'
        link = ET.SubElement(model, 'link', {'name': 'link'})

        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass').text = '0.1'
        inertia = ET.SubElement(inertial, 'inertia')
        for tag in ('ixx', 'iyy', 'izz'):
            ET.SubElement(inertia, tag).text = '0.001'
        for tag in ('ixy', 'ixz', 'iyz'):
            ET.SubElement(inertia, tag).text = '0'

        # A static weed needs no collision geometry: it never moves, and the
        # robot driving through a visual-only weed is exactly the behaviour we
        # want (real weeds don't stop a Thorvald, and the label pipeline
        # deliberately ignores them). Only the drop-and-settle fallback needs
        # a collider.
        if not static:
            collision = ET.SubElement(link, 'collision', {'name': 'collision'})
            coll_geom = ET.SubElement(collision, 'geometry')
            cylinder = ET.SubElement(coll_geom, 'cylinder')
            ET.SubElement(cylinder, 'radius').text = '0.12'
            ET.SubElement(cylinder, 'length').text = '0.3'

        visual = ET.SubElement(link, 'visual', {'name': 'visual'})
        vis_geom = ET.SubElement(visual, 'geometry')
        mesh = ET.SubElement(vis_geom, 'mesh')
        ET.SubElement(mesh, 'uri').text = WEED_MESH_URI
        ET.SubElement(mesh, 'scale').text = f'{scale:.5f} {scale:.5f} {scale:.5f}'
        material = ET.SubElement(visual, 'material')
        ET.SubElement(material, 'ambient').text = '0.15 0.35 0.05 1'
        ET.SubElement(material, 'diffuse').text = '0.25 0.55 0.08 1'
        ET.SubElement(material, 'specular').text = '0.05 0.05 0.05 1'

    return n_weeds


def pick_spawn(rows_file, rng, lateral_jitter_m=1.0, heading_jitter_deg=10.0):
    """Pick a random row, a random point along it, and a heading from the
    row's local tangent there -- then perturb both laterally and in
    heading, so row_follower has real correcting to do (spawning exactly
    on-row facing exactly along it would under-exercise it)."""
    with open(rows_file) as f:
        rows = yaml.safe_load(f)['rows']
    row = rng.choice(rows)
    pts = row['points']
    idx = rng.randrange(0, len(pts) - 1)
    x0, y0 = pts[idx]
    x1, y1 = pts[idx + 1]
    heading = math.atan2(y1 - y0, x1 - x0)

    lateral = rng.uniform(-lateral_jitter_m, lateral_jitter_m)
    perp = heading + math.pi / 2
    x = x0 + lateral * math.cos(perp)
    y = y0 + lateral * math.sin(perp)
    yaw = heading + math.radians(rng.uniform(-heading_jitter_deg, heading_jitter_deg))
    return x, y, yaw, row['id']


def pick_control_strategy(rng, args):
    """Randomly choose row_follower's control_strategy and that
    strategy's own parameters, so dataset-generation runs get genuinely
    different trajectory dynamics (not just different noise on the same
    law) -- see row_follower.py's module docstring for what each of the
    four strategies actually does differently. Every strategy's
    parameters are always populated (not just the active strategy's) so
    downstream consumers (run_params.json) don't have to special-case
    which fields exist. k_damping/stanley_k_damping are deliberately NOT
    randomized: they're the fix for pd's/stanley's documented
    delay-induced oscillation (see row_follower.py), so they stay at
    their validated values rather than risking reintroducing instability
    per-run.

    Strategy choice is STRATIFIED into two trajectory-quality buckets,
    not picked uniformly across all four: 'pd'/'pure_pursuit' are
    empirically well-behaved (small, bounded deviations once tracking --
    see row_follower.py's validation notes), while 'stanley' has a
    documented rough ~10-15s initial transient (heading swings up to
    ~130 deg before settling) and 'bangbang' is deliberately
    discontinuous/weaving throughout -- picking all four with equal
    probability would make the dataset roughly 50% erratic, which swamps
    the "mostly clean tracking, a deliberate minority of erratic/
    recovery scenarios for robustness" mix wanted for training.
    --erratic-fraction controls that minority's size (default 0.15). The
    small_deviation bucket also gets its own, tighter perturb_std range
    (--small-deviation-perturb-std-max, well under the general
    --perturb-std-max) so it stays cleanly "small deviation" rather than
    occasionally rolling a wobbly outlier from the same wide range the
    erratic bucket uses."""
    if rng.random() < args.erratic_fraction:
        quality = 'erratic'
        strategy = rng.choice(['stanley', 'bangbang'])
        perturb_std = rng.uniform(args.perturb_std_min, args.perturb_std_max)
    else:
        quality = 'small_deviation'
        strategy = rng.choice(['pd', 'pure_pursuit'])
        perturb_std = rng.uniform(args.perturb_std_min, args.small_deviation_perturb_std_max)

    params = {
        'control_strategy': strategy,
        'trajectory_quality': quality,
        'forward_speed': rng.uniform(args.forward_speed_min, args.forward_speed_max),
        'lookahead_m': rng.uniform(args.lookahead_m_min, args.lookahead_m_max),
        'k_offset': rng.uniform(args.k_offset_min, args.k_offset_max),
        'k_heading': rng.uniform(args.k_heading_min, args.k_heading_max),
        'k_stanley': rng.uniform(args.k_stanley_min, args.k_stanley_max),
        'stanley_v_epsilon': rng.uniform(args.stanley_v_epsilon_min, args.stanley_v_epsilon_max),
        'stanley_k_damping': args.stanley_k_damping,
        'bangbang_deadband_m': rng.uniform(args.bangbang_deadband_m_min, args.bangbang_deadband_m_max),
        'bangbang_magnitude': rng.uniform(args.bangbang_magnitude_min, args.bangbang_magnitude_max),
        'perturb_std': perturb_std,
    }
    return params


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows-file', required=True)
    parser.add_argument('--world-template', required=True)
    parser.add_argument('--out-world-dir', required=True)
    parser.add_argument('--session-name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--spawn-z', type=float, default=None,
                        help='spawn height; default: just above the terrain heightmap')
    parser.add_argument('--navigation-profile', choices=('diverse', 'expected'), default='diverse',
                        help="'expected' uses the empirically stable PD region and a small "
                             "initial displacement; 'diverse' keeps the recovery/erratic mix")
    # row_follower.py's 'pd' strategy now has a yaw-rate damping term
    # (k_damping) fixing the delay-induced oscillation that used to cap
    # this range low (see row_follower.py's module docstring) -- range
    # re-validated against that same previously-diverging perturb_std
    # before being widened back up here.
    parser.add_argument('--perturb-std-min', type=float, default=0.0)
    parser.add_argument('--perturb-std-max', type=float, default=0.08)
    # See pick_control_strategy's docstring -- keeps the ~85% "small
    # deviation" bucket meaningfully tighter than the erratic bucket's
    # full range, not just a different strategy on the same noise level.
    parser.add_argument('--small-deviation-perturb-std-max', type=float, default=0.03)
    parser.add_argument('--erratic-fraction', type=float, default=0.15,
                         help='Fraction of sessions drawn from the erratic bucket '
                              "(stanley/bangbang) rather than the small-deviation bucket "
                              '(pd/pure_pursuit). Default 0.15 = ~85%% small-deviation.')
    parser.add_argument('--lateral-jitter-m', type=float, default=1.0)
    parser.add_argument('--heading-jitter-deg', type=float, default=10.0)
    parser.add_argument('--forward-speed-min', type=float, default=0.15)
    parser.add_argument('--forward-speed-max', type=float, default=0.50)
    parser.add_argument('--lookahead-m-min', type=float, default=1.0)
    parser.add_argument('--lookahead-m-max', type=float, default=2.5)
    parser.add_argument('--k-offset-min', type=float, default=0.2)
    parser.add_argument('--k-offset-max', type=float, default=0.4)
    parser.add_argument('--k-heading-min', type=float, default=0.4)
    parser.add_argument('--k-heading-max', type=float, default=0.7)
    parser.add_argument('--camera-height-min', type=float, default=0.50)
    parser.add_argument('--camera-height-max', type=float, default=1.45)
    parser.add_argument('--camera-pitch-min', type=float, default=0.02)
    parser.add_argument('--camera-pitch-max', type=float, default=0.38)
    # A real mast-mounted camera is never square to the chassis. These are
    # small mounting misalignments, not behaviour: they change what the image
    # looks like (a tilted horizon, a rotated row) while the geometric ground
    # truth stays exact, because record_camera_poses composes and
    # export_recording inverts the SAME full rpy mount transform.
    # Roll is nearly free: it rotates the image about the optical axis, which
    # moves the row's vanishing point horizontally by only ~v_vp*sin(roll)
    # (~3 px at our pitch), so it adds appearance variation without corrupting
    # the target. Yaw is NOT free -- a mount yaw of psi shifts the row's
    # apparent direction by exactly psi, and psi is unobservable from a single
    # frame, so it becomes irreducible error on the heading label. At +/-0.05
    # rad that is ~1.65 deg std, ~19% of the heading signal's own spread; at
    # +/-0.015 it is ~0.5 deg (~6%). Kept small deliberately: this is meant to
    # model residual miscalibration, not to make the task unlearnable.
    parser.add_argument('--camera-roll-abs-max', type=float, default=0.06)    # ~3.4 deg
    parser.add_argument('--camera-yaw-abs-max', type=float, default=0.015)   # ~0.86 deg
    # Field of view varies the intrinsics themselves. Safe to randomize because
    # the label renderer reads fx/fy from the CameraInfo actually published by
    # the sensor, rather than assuming a fixed focal length. 1.05-1.75 rad is
    # roughly a 60-100 deg lens.
    parser.add_argument('--camera-hfov-min', type=float, default=1.05)
    parser.add_argument('--camera-hfov-max', type=float, default=1.75)
    # Sensor noise, applied by gz to the rendered RGB only -- masks are
    # projected geometrically and stay clean, which is what we want: the model
    # must learn the clean label from a noisy image.
    parser.add_argument('--camera-noise-max', type=float, default=0.012)
    # Forward overhang of the camera from base_link. On thorvald_ii_4wd2ws_tora
    # base_link sits on the front axle, so this is an unsupported lever arm:
    # the larger it is, the more chassis pitch turns into horizon swing and
    # vertical camera bob. Randomized as a dataset axis; the old fixed 0.80 was
    # sized for a different chassis and sits at the top of this range.
    parser.add_argument('--camera-x-min', type=float, default=0.25)
    parser.add_argument('--camera-x-max', type=float, default=0.80)
    parser.add_argument('--k-stanley-min', type=float, default=0.6)
    parser.add_argument('--k-stanley-max', type=float, default=1.0)
    parser.add_argument('--stanley-v-epsilon-min', type=float, default=0.3)
    parser.add_argument('--stanley-v-epsilon-max', type=float, default=0.6)
    # Not randomized, like pd's k_damping: this is the fix for stanley's
    # own documented rough-transient/oscillation issue (see
    # row_follower.py), not a behavioral knob -- stays at its validated
    # value rather than risking reintroducing instability per-run.
    parser.add_argument('--stanley-k-damping', type=float, default=0.6)
    parser.add_argument('--bangbang-deadband-m-min', type=float, default=0.05)
    parser.add_argument('--bangbang-deadband-m-max', type=float, default=0.2)
    parser.add_argument('--bangbang-magnitude-min', type=float, default=0.1)
    parser.add_argument('--bangbang-magnitude-max', type=float, default=0.25)
    parser.add_argument('--output-json', default=None,
                         help='Optional path to also write a structured run_params.json '
                              '(same values as the KEY="VALUE" stdout lines, machine-readable)')
    # Widened for the diversity campaign: denser clutter, and offsets reaching
    # 0.15 m so some weeds sit on the bed edge / in the robot's path rather than
    # only tidily in the corridor centre. The label pipeline is unchanged --
    # weeds are deliberately absent from the ground truth, so this trains
    # robustness to clutter against the TRUE row geometry.
    parser.add_argument('--terrain-heightmap', default='farm/fields/20250425/terrain_heightmap.npz',
                        help='baked heightmap (bake_terrain_heightmap.py) enabling STATIC weed '
                             'placement; without it weeds fall back to drop-and-settle')
    parser.add_argument('--n-weeds-min', type=int, default=0)
    parser.add_argument('--n-weeds-max', type=int, default=25)
    parser.add_argument('--weed-corridor-offset-min', type=float, default=0.15)
    parser.add_argument('--weed-corridor-offset-max', type=float, default=1.0)
    return parser


def default_args(**overrides):
    """Parser defaults as a Namespace, with keyword overrides (underscored names)."""
    required = ('rows_file', 'world_template', 'out_world_dir', 'session_name', 'seed')
    argv = []
    for name in required:
        argv += ['--' + name.replace('_', '-'), str(overrides.get(name, 0))]
    args = build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def generate(args):
    """Write the randomised world for one session and return its run parameters."""
    rng = random.Random(args.seed)

    os.makedirs(args.out_world_dir, exist_ok=True)
    world_path = os.path.join(args.out_world_dir, f'{args.session_name}.sdf')
    direction = randomize_world(args.world_template, world_path, rng)

    # Second pass over the just-written world file to scatter weeds --
    # kept separate from randomize_world (which owns lighting) rather
    # than merged into it, so each stays a focused, independently
    # testable transform of the world SDF.
    weed_tree = ET.parse(world_path)
    weed_world_elem = weed_tree.getroot().find('.//world')
    heightmap = load_heightmap(args.terrain_heightmap)
    n_weeds = add_weeds(weed_world_elem, args.rows_file, rng,
                         n_weeds_range=(args.n_weeds_min, args.n_weeds_max),
                         corridor_offset_range=(args.weed_corridor_offset_min, args.weed_corridor_offset_max),
                         heightmap=heightmap)
    weed_tree.write(world_path, encoding='utf-8', xml_declaration=True)

    if args.navigation_profile == 'expected':
        # Measured over the existing 497-session campaign: PD runs with
        # |initial offset| <= 0.25 m passed the tail-half tracking criterion
        # 88.5% of the time; higher k_offset/k_heading thirds were the most
        # reliable.  This profile creates clean nominal-navigation sequences,
        # while the default profile remains available for recovery diversity.
        spawn_lateral_jitter = min(args.lateral_jitter_m, 0.25)
        spawn_heading_jitter = min(args.heading_jitter_deg, 3.0)
    else:
        spawn_lateral_jitter = args.lateral_jitter_m
        spawn_heading_jitter = args.heading_jitter_deg
    x, y, yaw, row_id = pick_spawn(
        args.rows_file, rng, spawn_lateral_jitter, spawn_heading_jitter)
    control_params = pick_control_strategy(rng, args)
    if args.navigation_profile == 'expected':
        control_params.update({
            'control_strategy': 'pd',
            'trajectory_quality': 'expected_navigation',
            'forward_speed': rng.uniform(0.20, 0.35),
            'k_offset': rng.uniform(0.35, 0.40),
            'k_heading': rng.uniform(0.62, 0.70),
            'perturb_std': rng.uniform(0.0, 0.01),
        })
    camera_height = rng.uniform(args.camera_height_min, args.camera_height_max)
    camera_pitch = rng.uniform(args.camera_pitch_min, args.camera_pitch_max)
    camera_x = rng.uniform(args.camera_x_min, args.camera_x_max)
    camera_roll = rng.uniform(-args.camera_roll_abs_max, args.camera_roll_abs_max)
    camera_yaw = rng.uniform(-args.camera_yaw_abs_max, args.camera_yaw_abs_max)
    camera_hfov = rng.uniform(args.camera_hfov_min, args.camera_hfov_max)
    if args.navigation_profile == 'expected':
        # This profile is also intended to replenish fully visible line
        # targets. For the 1 m near point, a large camera overhang/high mount
        # puts the point below the image even during perfect tracking. These
        # bounds keep both 1 m and 5 m control points in a 640x480 view on flat
        # ground across the whole range (with margin for terrain attitude).
        camera_height = rng.uniform(0.65, 0.75)
        camera_pitch = rng.uniform(0.28, 0.35)
        camera_x = rng.uniform(0.25, 0.30)
        camera_hfov = rng.uniform(1.50, 1.70)
    # Half the sessions stay noise-free so the dataset keeps a clean subset.
    camera_noise = 0.0 if rng.random() < 0.5 else rng.uniform(0.002, args.camera_noise_max)

    return {
        'seed': args.seed,
        'session_name': args.session_name,
        'world_path': world_path,
        'spawn': {'x': x, 'y': y, 'yaw': yaw,
                  'z': args.spawn_z if args.spawn_z is not None else spawn_height(heightmap, x, y)},
        'row_id': row_id,
        'navigation_profile': args.navigation_profile,
        'sun_direction': list(direction),
        'camera_height': camera_height,
        'camera_pitch': camera_pitch,
        'camera_x': camera_x,
        'camera_roll': camera_roll,
        'camera_yaw': camera_yaw,
        'camera_hfov': camera_hfov,
        'camera_noise_stddev': camera_noise,
        'n_weeds': n_weeds,
        **control_params,
    }


def main():
    args = build_parser().parse_args()
    params = generate(args)
    flat = {'WORLD_PATH': params['world_path'], 'SPAWN_X': params['spawn']['x'],
            'SPAWN_Y': params['spawn']['y'], 'SPAWN_Z': params['spawn']['z'],
            'SPAWN_YAW': params['spawn']['yaw'], 'ROW_ID': params['row_id'],
            'SUN_DIRECTION': ' '.join(f'{v:.4f}' for v in params['sun_direction'])}
    for key in ('perturb_std', 'control_strategy', 'trajectory_quality', 'forward_speed',
                'lookahead_m', 'k_offset', 'k_heading', 'k_stanley', 'stanley_v_epsilon',
                'stanley_k_damping', 'bangbang_deadband_m', 'bangbang_magnitude',
                'camera_height', 'camera_pitch', 'camera_x', 'camera_roll', 'camera_yaw',
                'camera_hfov', 'camera_noise_stddev', 'n_weeds'):
        flat[key.upper().replace('CAMERA_NOISE_STDDEV', 'CAMERA_NOISE')] = params[key]
    for key, value in flat.items():
        print(f'{key}="{value}"')
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(params, f, indent=2)


if __name__ == '__main__':
    main()
