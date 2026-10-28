"""Simulation bringup for thorvald_ii_4wd4ws_std."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_model  = get_package_share_directory('thorvald_model')
    pkg_teleop = get_package_share_directory('thorvald_teleop')
    pkg_mux    = get_package_share_directory('thorvald_twist_mux')
    pkg_sim    = get_package_share_directory('thorvald_sim')

    model_cfg = os.path.join(pkg_model, 'config', 'thorvald_ii_4wd4ws_std', 'thorvald_ii_4wd4ws_std.yaml')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_sim, 'launch', 'sim.launch.py')),
        launch_arguments={
            'robot_model': model_cfg,
        }.items())

    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_teleop, 'launch', 'teleop_logitech_f710.launch.py')))

    mux_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_mux, 'launch', 'twist_mux.launch.py')))

    return LaunchDescription([
        sim_launch,
        teleop_launch,
        mux_launch,
    ])
