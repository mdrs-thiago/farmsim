"""One-time capture of a top-down field image for the row-marking step.

Loads the farm world (no birdseye camera baked in -- see
worlds/birdseye_camera.sdf for why: it used to live directly in
vollebekk_20250624.sdf, but collided with the robot's own front_camera on
gz-sim's shared /camera_info topic whenever farm_sim.launch.py spawned
the robot into that same world file, silently corrupting the front
camera's intrinsics), then spawns worlds/birdseye_camera.sdf into it
dynamically -- so it only ever exists for this capture-only launch, never
alongside the robot.

Camera pose (cam_x/y/z/yaw) and resolution (image_size/far_clip) are
launch args, not hardcoded, since getting a good top-down shot always
takes at least two passes per map: a first wide/coarse look to measure
the crop's actual extent and heading (see segment_rows.field_rotated_rect
and the history note in worlds/birdseye_camera.sdf), then a tight,
rotation-aligned, high-resolution second pass framed from that
measurement. Defaults are the 20250624 field's already-tuned final-pass
values; every other map needs its own two-pass run with different values.

output_tag defaults to whatever's between 'vollebekk_' and '.sdf' in the
world filename (e.g. vollebekk_20250425.sdf -> '20250425'), so output
filenames (topdown_<tag>.png/.yaml) can't silently drift out of sync with
which world was actually captured, or overwrite another date's capture.
"""

import os

from ament_index_python.packages import get_package_share_directory
from farmsim_farm.fields import model_resource_path
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _setup(context: LaunchContext):
    pkg_share = get_package_share_directory('farmsim_farm')

    world = context.perform_substitution(LaunchConfiguration('world'))
    output_tag = context.perform_substitution(LaunchConfiguration('output_tag'))
    if not output_tag:
        base = os.path.splitext(os.path.basename(world))[0]
        output_tag = base[len('vollebekk_'):] if base.startswith('vollebekk_') else base

    template_path = os.path.join(pkg_share, 'worlds', 'birdseye_camera.sdf')
    with open(template_path, 'r') as f:
        template = f.read()
    sdf_string = template.format(
        image_size=context.perform_substitution(LaunchConfiguration('image_size')),
        far_clip=context.perform_substitution(LaunchConfiguration('far_clip')),
    )

    # Delayed so gz-sim's entity-creation service is definitely up first --
    # unlike the -topic based robot spawn elsewhere (which just waits for a
    # ROS topic message and so has no such race), -string spawns call the
    # creation service directly and don't retry if it isn't ready yet.
    #
    # Pose passed explicitly via -x/-y/-z/-R/-P/-Y -- confirmed empirically
    # that `create` does NOT apply a spawned model's embedded <pose> the
    # way a world file's own <include><pose> would; the first attempt at
    # this spawned the camera at the origin with no rotation applied (a
    # sideways, roughly-level view of the terrain, not top-down) despite
    # the SDF having the intended pose right there in it.
    spawn_birdseye_camera = TimerAction(
        period=3.0,
        actions=[Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_birdseye_camera',
            output='screen',
            arguments=[
                '-string', sdf_string, '-name', 'birdseye_camera',
                '-x', context.perform_substitution(LaunchConfiguration('cam_x')),
                '-y', context.perform_substitution(LaunchConfiguration('cam_y')),
                '-z', context.perform_substitution(LaunchConfiguration('cam_z')),
                '-R', '0', '-P', '1.5708',
                '-Y', context.perform_substitution(LaunchConfiguration('cam_yaw')),
            ],
        )],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/birdseye_camera@sensor_msgs/msg/Image[gz.msgs.Image',
            # Same gz-sim quirk as the front camera: CameraInfo lands on a
            # flat /camera_info topic regardless of the sensor's <topic>.
            # Harmless here since this launch never has more than one
            # camera in the world -- see birdseye_camera.sdf for why that
            # matters when it isn't true (farm_sim.launch.py).
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[('/camera_info', '/birdseye_camera/camera_info')],
        output='screen',
    )

    capture = Node(
        package='farmsim_farm',
        executable='capture_topdown',
        name='capture_topdown',
        output='screen',
        parameters=[{
            'output_dir': context.perform_substitution(LaunchConfiguration('output_dir')),
            'output_tag': output_tag,
            'camera_x': float(context.perform_substitution(LaunchConfiguration('cam_x'))),
            'camera_y': float(context.perform_substitution(LaunchConfiguration('cam_y'))),
            'camera_z': float(context.perform_substitution(LaunchConfiguration('cam_z'))),
            'camera_roll': 0.0,
            'camera_pitch': 1.5708,
            'camera_yaw': float(context.perform_substitution(LaunchConfiguration('cam_yaw'))),
            'ground_plane_z': float(context.perform_substitution(LaunchConfiguration('ground_plane_z'))),
        }],
    )

    print(f"Capturing world='{world}' -> topdown_{output_tag}.png/.yaml")
    return [spawn_birdseye_camera, bridge, capture]


def generate_launch_description():
    pkg_share = get_package_share_directory('farmsim_farm')
    default_world = os.path.join(pkg_share, 'worlds', 'vollebekk_20250624.sdf')

    world_arg = DeclareLaunchArgument(
        'world', default_value=default_world,
        description='Path to the Gazebo (gz-sim) farm world file',
    )
    models_dir_arg = DeclareLaunchArgument(
        'models_dir', default_value=model_resource_path(),
        description='GZ_SIM_RESOURCE_PATH entries holding the field models (farm/fields/*/model)',
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='true',
        description="Run gz sim server-only, no GUI window (default true -- this "
                    "launch only needs to grab one frame).",
    )
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/run/media/thiane/59B6290966A08007/projects/nmbu/thorvald_ros2_ws/captures',
        description='Directory to write the captured PNG + camera pose/intrinsics YAML',
    )
    output_tag_arg = DeclareLaunchArgument(
        'output_tag', default_value='',
        description='Filename tag for topdown_<tag>.png/.yaml; empty = derive from the '
                    "world filename (vollebekk_<tag>.sdf).",
    )
    cam_x_arg = DeclareLaunchArgument('cam_x', default_value='-36.4446')
    cam_y_arg = DeclareLaunchArgument('cam_y', default_value='1.7231')
    cam_z_arg = DeclareLaunchArgument(
        'cam_z', default_value='39',
        description='Camera altitude, meters -- with the fixed 90deg horizontal_fov, '
                    'ground footprint width = 2 * cam_z (straight-down pitch).',
    )
    cam_yaw_arg = DeclareLaunchArgument(
        'cam_yaw', default_value='0.51245',
        description='Camera yaw, radians -- 0 for an axis-aligned first pass over a new '
                    "map, or the crop's measured long-axis heading for a tight second pass.",
    )
    image_size_arg = DeclareLaunchArgument(
        'image_size', default_value='6400',
        description='Square image width=height in pixels. Effective resolution = '
                    '(2*cam_z) / image_size m/px.',
    )
    far_clip_arg = DeclareLaunchArgument(
        'far_clip', default_value='200',
        description='Camera far clip plane, meters -- must exceed cam_z plus a margin.',
    )
    ground_plane_z_arg = DeclareLaunchArgument(
        'ground_plane_z', default_value='-0.8475',
        description='Assumed flat-ground elevation for this map, written into the '
                    'metadata sidecar for pixel->world math (mark_rows.py/segment_rows.py). '
                    "Use this map's own median terrain z, not 20250624's -- photogrammetry "
                    'meshes commonly have a handful of high outlier vertices, so prefer a '
                    'percentile/median estimate over the raw min/max.',
    )

    models_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', LaunchConfiguration('models_dir'))

    gz_partition = SetEnvironmentVariable('GZ_PARTITION', 'thorvald_topdown_capture')

    headless_flag = PythonExpression([
        "'-s --headless-rendering ' if '", LaunchConfiguration('headless'), "' == 'true' else ''",
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': [headless_flag, '-r ', LaunchConfiguration('world')],
        }.items(),
    )

    return LaunchDescription([
        world_arg, models_dir_arg, headless_arg, output_dir_arg, output_tag_arg,
        cam_x_arg, cam_y_arg, cam_z_arg, cam_yaw_arg, image_size_arg, far_clip_arg,
        ground_plane_z_arg,
        gz_partition,
        models_resource_path,
        gz_sim,
        OpaqueFunction(function=_setup),
    ])
