"""Spawn the Thorvald II model into a surveyed farm field on gz-sim.

  - robot: thorvald_model with control_backend='gz_sim' and the front camera
    (gz_camera.xacro); thorvald_base's twist_to_joints (real kinematics)
    bridged to the gz joint plugins by thorvald_gz's joint_command_splitter.
  - farm: worlds/vollebekk_<field>.sdf; the photogrammetry meshes in
    farm/fields/<field>/model are found through GZ_SIM_RESOURCE_PATH.
  - control (control:=true): farmsim_control's controller_node driving the
    path in path_file, on ground truth or on the CNN (pose_source).
  - perception (perception:=true): farmsim_perception's perception_node.

Recording a ros2 bag of the camera (+ tf/odom) is optional (record:=true).
"""

import os
from xml.dom import Node as DomNode

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_context import LaunchContext
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from farmsim_farm.fields import FIELDS_DIR, WORLDS_DIR, model_resource_path

BASE_BRIDGE_ARGS = [
    '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
    '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
    '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
    '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
    '/front_camera@sensor_msgs/msg/Image[gz.msgs.Image',
    # gz-sim publishes this camera's CameraInfo on a flat /camera_info
    # topic regardless of the sensor's configured <topic> (image respects
    # it, camera_info does not) -- bridged as-is and remapped below.
    '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    # only published when model_extras is gz_camera_chase.xacro (run.video)
    '/chase_camera@sensor_msgs/msg/Image[gz.msgs.Image',
]


def _joint_bridge_args(num_drives):
    return [
        f'/steering{i}/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double'
        for i in range(num_drives)
    ] + [
        f'/wheel{i}/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double'
        for i in range(num_drives)
    ]


def _count_drives(robot_config):
    """Number of drive modules in a robot_config yaml (1 for the tricycle
    variant, 4 for the standard 4wd4ws, up to 9 for 'crazy', etc.) -- same
    counting logic as thorvald_base's twist_to_joints.py, needed here to
    size the per-joint bridge topic list to whatever --robot_config is
    passed in, not just the default 4wd4ws config."""
    with open(robot_config, 'r') as f:
        motor_drives = yaml.safe_load(f).get('motor_drives', {})
    n = 0
    while f'drive{n}' in motor_drives:
        n += 1
    return n


def _remove_comments_and_blank_text(node):
    for child in list(node.childNodes):
        if child.nodeType == DomNode.COMMENT_NODE:
            node.removeChild(child)
        elif child.nodeType == DomNode.TEXT_NODE and not child.data.strip():
            node.removeChild(child)
        else:
            _remove_comments_and_blank_text(child)


def _robot_description(robot_config, model_extras, camera_height, camera_pitch, camera_x,
                       camera_roll, camera_yaw, camera_hfov, camera_noise_stddev):
    thorvald_model_share = get_package_share_directory('thorvald_model')
    xacro_file = os.path.join(thorvald_model_share, 'urdf', 'thorvald_model.xacro')
    mappings = {
        'robot_name': 'thorvald',
        'tf_prefix': '',
        'robot_config': robot_config,
        'model_extras': model_extras,
        'control_backend': 'gz_sim',
        'camera_height': camera_height,
        'camera_pitch': camera_pitch,
        'camera_x': camera_x,
        'camera_roll': camera_roll,
        'camera_yaw': camera_yaw,
        'camera_hfov': camera_hfov,
        'camera_noise_stddev': camera_noise_stddev,
    }
    doc = xacro.process_file(xacro_file, mappings=mappings)
    _remove_comments_and_blank_text(doc)
    description = doc.documentElement.toxml()

    # thorvald_model.xacro's meshes all use package:// URIs (legs, towers,
    # battery, corner brackets). gz-sim's mesh loader needs a real
    # filesystem/file:// URI (unlike nav_control's hand-built model, which
    # only uses primitive geometry and never exercises this path), so patch
    # the URIs to the package's actual share directory before spawning.
    description = description.replace(
        'package://thorvald_model/', f'file://{thorvald_model_share}/')
    return description


def spawn_robot(context: LaunchContext, x, y, z, yaw):
    robot_config = context.perform_substitution(LaunchConfiguration('robot_config'))
    model_extras = context.perform_substitution(LaunchConfiguration('model_extras'))
    camera_height = context.perform_substitution(LaunchConfiguration('camera_height'))
    camera_pitch = context.perform_substitution(LaunchConfiguration('camera_pitch'))
    camera_x = context.perform_substitution(LaunchConfiguration('camera_x'))
    camera_roll = context.perform_substitution(LaunchConfiguration('camera_roll'))
    camera_yaw = context.perform_substitution(LaunchConfiguration('camera_yaw'))
    camera_hfov = context.perform_substitution(LaunchConfiguration('camera_hfov'))
    camera_noise_stddev = context.perform_substitution(LaunchConfiguration('camera_noise_stddev'))

    robot_description = _robot_description(
        robot_config, model_extras, camera_height, camera_pitch, camera_x,
        camera_roll, camera_yaw, camera_hfov, camera_noise_stddev)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_description},
        ],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        name='ros_gz_sim_create',
        output='both',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'thorvald',
            '-x', context.perform_substitution(x),
            '-y', context.perform_substitution(y),
            '-z', context.perform_substitution(z),
            '-Y', context.perform_substitution(yaw),
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        arguments=BASE_BRIDGE_ARGS + _joint_bridge_args(_count_drives(robot_config)),
        parameters=[
            {'qos_overrides./tf_static.publisher.durability': 'transient_local'},
        ],
        remappings=[('/camera_info', '/front_camera/camera_info')],
        output='screen',
    )

    twist_to_joints = Node(
        package='thorvald_base',
        executable='twist_to_joints',
        name='twist_to_joints',
        output='screen',
        parameters=[{
            'robot_config': robot_config,
            'max_steering_rate': 1.5,
        }],
    )

    joint_command_splitter = Node(
        package='thorvald_gz',
        executable='joint_command_splitter',
        name='joint_command_splitter',
        output='screen',
        parameters=[{'robot_config': robot_config}],
    )

    controller = Node(
        package='farmsim_control',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'config_file': context.perform_substitution(LaunchConfiguration('control_config')),
            'path_file': context.perform_substitution(LaunchConfiguration('path_file')),
            'pose_source': context.perform_substitution(LaunchConfiguration('pose_source')),
            'log_file': context.perform_substitution(LaunchConfiguration('control_log')),
        }],
        condition=IfCondition(LaunchConfiguration('control')),
    )

    perception = Node(
        package='farmsim_perception',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{
            'checkpoint': context.perform_substitution(LaunchConfiguration('checkpoint')),
            'show': context.perform_substitution(LaunchConfiguration('perception_view')) == 'true',
        }],
        condition=IfCondition(LaunchConfiguration('perception')),
    )

    field = context.perform_substitution(LaunchConfiguration('field'))
    rows_file = os.path.join(FIELDS_DIR, field, 'rows_centerline.yaml')

    live_segmentation_view = Node(
        package='farmsim_perception',
        executable='live_segmentation_view',
        name='live_segmentation_view',
        output='screen',
        condition=IfCondition(LaunchConfiguration('live_view')),
    )

    # Publishes /cmd_vel from keyboard input in its own OpenCV window --
    # don't run alongside control:=true, both would publish /cmd_vel.
    live_label_teleop = Node(
        package='farmsim_perception',
        executable='live_label_teleop',
        name='live_label_teleop',
        output='screen',
        parameters=[{
            'rows_file': rows_file,
            'camera_height': float(camera_height),
            'camera_pitch': float(camera_pitch),
            'camera_x_offset': float(camera_x),
            'topdown_meta_file': os.path.join(FIELDS_DIR, field, 'topdown.yaml'),
        }],
        condition=IfCondition(LaunchConfiguration('teleop_view')),
    )

    return [
        robot_state_publisher, spawn, bridge, twist_to_joints, joint_command_splitter,
        controller, perception, live_segmentation_view, live_label_teleop,
    ]


def _default_world(context):
    field = context.perform_substitution(LaunchConfiguration('field'))
    world = context.perform_substitution(LaunchConfiguration('world')) or str(WORLDS_DIR / f'vollebekk_{field}.sdf')
    return world


def generate_launch_description():
    field_arg = DeclareLaunchArgument(
        'field', default_value='20250425',
        description='Survey date of the field (folder name in farm/fields/)')
    world_arg = DeclareLaunchArgument(
        'world', default_value='',
        description='World SDF; empty = worlds/vollebekk_<field>.sdf. record_dataset.py passes '
                    'a lighting/weed-randomised copy here.',
    )
    models_dir_arg = DeclareLaunchArgument(
        'models_dir', default_value=model_resource_path(),
        description='GZ_SIM_RESOURCE_PATH entries holding the field models (farm/fields/*/model)',
    )
    robot_config_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('thorvald_model'),
            'config', 'thorvald_ii_4wd2ws_tora', 'thorvald_ii_4wd2ws_tora.yaml',
        ]),
        description='Absolute path to the robot YAML configuration file',
    )
    model_extras_arg = DeclareLaunchArgument(
        'model_extras',
        default_value=PathJoinSubstitution([
            FindPackageShare('thorvald_model'),
            'urdf', 'sensor_modules', 'gz_camera.xacro',
        ]),
        description='Path to the xacro extras file (sensors); default adds the front camera',
    )
    camera_height_arg = DeclareLaunchArgument(
        'camera_height', default_value='0.90',
        description='Front camera mount height above base_link, in meters '
                    '(only used when model_extras is gz_camera.xacro)',
    )
    camera_pitch_arg = DeclareLaunchArgument(
        'camera_pitch', default_value='0.15',
        description='Front camera nose-down tilt, in radians, positive = more downward '
                    '(only used when model_extras is gz_camera.xacro)',
    )
    camera_x_arg = DeclareLaunchArgument(
        'camera_x', default_value='0.80',
        description='Front camera forward mount offset from base_link, in meters '
                    '(only used when model_extras is gz_camera.xacro). Must match '
                    "record_camera_poses' camera_x_offset.",
    )
    camera_roll_arg = DeclareLaunchArgument(
        'camera_roll', default_value='0.0',
        description='Front camera mount roll, radians (small mounting misalignment)')
    camera_yaw_arg = DeclareLaunchArgument(
        'camera_yaw', default_value='0.0',
        description='Front camera mount yaw, radians (small mounting misalignment)')
    camera_hfov_arg = DeclareLaunchArgument(
        'camera_hfov', default_value='1.396',
        description='Camera horizontal field of view, radians; reaches the label '
                    'pipeline via the published CameraInfo (fx/fy)')
    camera_noise_stddev_arg = DeclareLaunchArgument(
        'camera_noise_stddev', default_value='0.0',
        description='Gaussian sensor noise stddev on the rendered image (0 = clean)')
    x_arg = DeclareLaunchArgument('x', default_value='0.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument(
        'z', default_value='2.0',
        description='Spawn height -- terrain ground level at (x,y) is not known in '
                     'advance for the photogrammetry meshes, so the robot is dropped '
                     'from a small height. Tune once you see where it lands.',
    )
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')
    control_arg = DeclareLaunchArgument(
        'control', default_value='false',
        description='Run farmsim_control controller_node along path_file.')
    control_config_arg = DeclareLaunchArgument(
        'control_config', default_value='',
        description='YAML with a control: block (cfg/control.yaml after overrides).')
    path_file_arg = DeclareLaunchArgument(
        'path_file', default_value='', description='Row-pass path YAML (cfg/paths/*.yaml).')
    pose_source_arg = DeclareLaunchArgument(
        'pose_source', default_value='gt',
        description="Row state for the in-row controller: 'gt' or 'perception'.")
    control_log_arg = DeclareLaunchArgument(
        'control_log', default_value='', description='JSON-lines log written by controller_node.')
    perception_arg = DeclareLaunchArgument(
        'perception', default_value='false', description='Run farmsim_perception perception_node.')
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint', default_value='', description='row_net checkpoint for perception_node.')
    perception_view_arg = DeclareLaunchArgument(
        'perception_view', default_value='false',
        description='OpenCV window with the CNN segmentation and row line.')
    live_view_arg = DeclareLaunchArgument(
        'live_view', default_value='false',
        description='Open a live OpenCV window showing the front camera feed with the '
                    'crop-row segmentation overlay, updated every frame, alongside the '
                    'Gazebo GUI (requires a display; needs the same X server headless:=false uses).',
    )
    teleop_view_arg = DeclareLaunchArgument(
        'teleop_view', default_value='false',
        description='Open a live OpenCV window showing the front camera feed with the '
                    "field's row centerline(s) projected onto it, and drive the robot "
                    'by keyboard (w/a/s/d) from that same window. Requires control:=false '
                    '(both publish /cmd_vel otherwise).',
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description="Run gz sim server-only, no GUI window.",
    )
    record_arg = DeclareLaunchArgument(
        'record', default_value='false',
        description='Record the front camera (+ tf/odom) to a ros2 bag.',
    )
    record_path_arg = DeclareLaunchArgument(
        'record_path',
        default_value='outputs/recordings/farm_run_bag',
        description="Output directory for the ros2 bag (must not already exist -- "
                    "override per run, e.g. record_path:=.../farm_run_2).",
    )

    models_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', LaunchConfiguration('models_dir'))

    # gz-transport's default partition (when GZ_PARTITION is unset) is
    # shared by every gz-sim process on the machine, so topics as generic
    # as /clock, /odom, /tf collide with any other gz-sim world already
    # running (e.g. nav_control's wheat_field_sim). Giving this launch its
    # own partition keeps it isolated regardless of what else is running.
    gz_partition = SetEnvironmentVariable('GZ_PARTITION', 'farmsim')

    headless_flag = PythonExpression([
        "'-s --headless-rendering ' if '", LaunchConfiguration('headless'), "' == 'true' else ''",
    ])

    def gz_sim(context):
        return [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
            launch_arguments={
                'gz_args': context.perform_substitution(headless_flag) + '-r ' + _default_world(context),
            }.items(),
        )]

    spawn = OpaqueFunction(
        function=spawn_robot,
        args=[
            LaunchConfiguration('x'),
            LaunchConfiguration('y'),
            LaunchConfiguration('z'),
            LaunchConfiguration('yaw'),
        ],
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', LaunchConfiguration('record_path'),
            '--use-sim-time',
            '--topics', '/front_camera', '/front_camera/camera_info', '/tf', '/odom',
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('record')),
    )

    return LaunchDescription([
        field_arg, world_arg, models_dir_arg, robot_config_arg, model_extras_arg,
        camera_height_arg, camera_pitch_arg, camera_x_arg,
        camera_roll_arg, camera_yaw_arg, camera_hfov_arg, camera_noise_stddev_arg,
        x_arg, y_arg, z_arg, yaw_arg,
        control_arg, control_config_arg, path_file_arg, pose_source_arg, control_log_arg,
        perception_arg, checkpoint_arg, perception_view_arg,
        live_view_arg, teleop_view_arg,
        headless_arg, record_arg, record_path_arg,
        gz_partition,
        models_resource_path,
        OpaqueFunction(function=gz_sim),
        spawn,
        bag_record,
    ])
