"""
sim.launch.py  —  Thorvald II Gazebo Classic simulation (ROS2 Humble)

Launches:
  1. gzserver + gzclient  (Gazebo 11 Classic)
  2. robot_state_publisher  (URDF → /robot_description, TF)
  3. spawn_entity           (spawn robot in Gazebo from /robot_description)
  4. joint_state_broadcaster  (after spawn)
  5. wheel_velocity_controller + steering_position_controller  (after broadcaster)
  6. twist_to_joints         (kinematics: /cmd_vel → controller commands)

Usage:
  ros2 launch thorvald_sim sim.launch.py
  ros2 launch thorvald_sim sim.launch.py robot_model:=<path_to_yaml>
  ros2 launch thorvald_sim sim.launch.py gui:=false   # headless
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_thorvald_model = get_package_share_directory('thorvald_model')
    thorvald_model_parent = os.path.dirname(pkg_thorvald_model)

    gazebo_model_paths = [
        path for path in os.environ.get('GAZEBO_MODEL_PATH', '').split(os.pathsep)
        if path and os.path.normpath(path) != '/opt/ros/humble/share'
    ]
    gazebo_model_path = os.pathsep.join([thorvald_model_parent] + gazebo_model_paths)
    gazebo_model_path_env = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=gazebo_model_path,
    )

    # ------------------------------------------------------------------ args
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='thorvald',
        description='Name used when spawning the robot entity in Gazebo',
    )
    tf_prefix_arg = DeclareLaunchArgument(
        'tf_prefix',
        default_value='',
        description='Prefix prepended to all TF frame names',
    )
    robot_model_arg = DeclareLaunchArgument(
        'robot_model',
        default_value=PathJoinSubstitution([
            FindPackageShare('thorvald_model'),
            'config', 'thorvald_ii_4wd4ws_std', 'thorvald_ii_4wd4ws_std.yaml',
        ]),
        description='Absolute path to the robot YAML configuration file',
    )
    model_extras_arg = DeclareLaunchArgument(
        'model_extras',
        default_value='',
        description=(
            'Path to an optional xacro extras file (sensors, attachments). '
            'Leave empty to skip.'
        ),
    )
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Set to "false" to run Gazebo headless (gzserver only)',
    )
    max_steering_rate_arg = DeclareLaunchArgument(
        'max_steering_rate',
        default_value='1.5',
        description='Maximum steering angle change rate in rad/s',
    )

    # ------------------------------------------------------ Gazebo Classic
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'verbose': 'true',
            'gui': LaunchConfiguration('gui'),
        }.items(),
    )

    # ----------------------------------------- robot_state_publisher (URDF)
    # Note: does NOT set use_sim_time here — robot_state_publisher only
    # publishes /robot_description and static TFs, so wall-clock is fine.
    model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('thorvald_model'), 'launch', 'model.launch.py',
            ])
        ),
        launch_arguments={
            'robot_name':   LaunchConfiguration('robot_name'),
            'tf_prefix':    LaunchConfiguration('tf_prefix'),
            'robot_model':  LaunchConfiguration('robot_model'),
            'model_extras': LaunchConfiguration('model_extras'),
        }.items(),
    )

    # -------------------------------------------- spawn robot in Gazebo
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', LaunchConfiguration('robot_name'),
            '-timeout', '30',
        ],
        output='screen',
    )

    # ------------------------------------------ ros2_control spawners
    # use_sim_time is set explicitly on each controller node so that the
    # global SetParameter (which would affect even spawner processes) is
    # not needed.
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    wheel_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'wheel_velocity_controller',
            '--controller-manager', '/controller_manager',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    steering_position_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'steering_position_controller',
            '--controller-manager', '/controller_manager',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ------------------------------------------- kinematics node
    twist_to_joints = Node(
        package='thorvald_base',
        executable='twist_to_joints',
        parameters=[{
            'robot_config': LaunchConfiguration('robot_model'),
            'max_steering_rate': LaunchConfiguration('max_steering_rate'),
            'use_sim_time': True,
        }],
        output='screen',
    )

    # ------ event-driven sequencing ----------------------------------------
    # Step 1: spawn_robot exits → activate all controllers + kinematics
    delay_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[
                joint_state_broadcaster,
                wheel_velocity_controller,
                steering_position_controller,
                twist_to_joints,
            ],
        )
    )

    return LaunchDescription([
        # Args
        robot_name_arg,
        tf_prefix_arg,
        robot_model_arg,
        model_extras_arg,
        gui_arg,
        max_steering_rate_arg,
        # Let Gazebo resolve model://thorvald_model/... mesh URIs without
        # scanning every ROS package in /opt/ros/humble/share.
        gazebo_model_path_env,
        # Gazebo server + client
        gazebo,
        # Robot description publisher (runs independently of Gazebo timing)
        model_launch,
        # Spawn robot (waits until /spawn_entity service is available)
        spawn_robot,
        # Controllers + kinematics (triggered when spawn_entity exits cleanly)
        delay_after_spawn,
    ])
