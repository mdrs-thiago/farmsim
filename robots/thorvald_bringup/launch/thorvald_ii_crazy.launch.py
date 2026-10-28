"""Real-hardware bringup for thorvald_ii_crazy."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_model   = get_package_share_directory('thorvald_model')
    pkg_teleop  = get_package_share_directory('thorvald_teleop')
    pkg_mux     = get_package_share_directory('thorvald_twist_mux')
    pkg_bringup = get_package_share_directory('thorvald_bringup')

    model_cfg = os.path.join(pkg_model, 'config', 'thorvald_ii_crazy', 'thorvald_ii_crazy.yaml')

    model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_model, 'launch', 'model.launch.py')),
        launch_arguments={
            'robot_model': model_cfg,
        }.items())

    # Base driver loads the same robot YAML for drive params
    from launch_ros.actions import Node
    base_driver_node = Node(
        package='thorvald_base_driver',
        executable='base_driver',
        name='base_driver',
        arguments=['--ros-args', '--params-file', model_cfg],
        output='screen',
    )

    serial_node = Node(
        package='thorvald_io_serial',
        executable='thorvald_io_arduino',
        name='thorvald_io_arduino',
        output='screen',
    )

    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_teleop, 'launch', 'teleop_logitech_f710.launch.py')))

    mux_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mux, 'launch', 'twist_mux.launch.py')))

    return LaunchDescription([
        model_launch,
        base_driver_node,
        serial_node,
        teleop_launch,
        mux_launch,
    ])
