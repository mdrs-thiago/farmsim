"""
Master bringup launch file.
Accepts robot_variant and sim arguments; includes the appropriate variant launch.
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

SUPPORTED_VARIANTS = [
    'thorvald_ii_4wd4ws_std',
    'thorvald_ii_4wd4ws_slim',
    'thorvald_ii_6wd6ws',
    'thorvald_ii_4wd2ws_tora',
    'thorvald_ii_2wd0ws_tora',
    'thorvald_ii_1wd1ws_tric',
    'thorvald_ii_guren_ii',
    'thorvald_ii_crazy',
]


def _launch_setup(context, *args, **kwargs):
    variant = LaunchConfiguration('robot_variant').perform(context)
    sim     = LaunchConfiguration('sim').perform(context).lower() in ('true', '1', 'yes')

    if variant not in SUPPORTED_VARIANTS:
        raise RuntimeError(
            f'Unknown robot_variant: {variant!r}. '
            f'Supported: {SUPPORTED_VARIANTS}')

    pkg_bringup = get_package_share_directory('thorvald_bringup')
    suffix = '_sim' if sim else ''
    launch_file = os.path.join(pkg_bringup, 'launch', f'{variant}{suffix}.launch.py')

    if not os.path.exists(launch_file):
        raise RuntimeError(f'Launch file not found: {launch_file}')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file))
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_variant',
                              default_value='thorvald_ii_4wd4ws_std',
                              description='Robot platform variant'),
        DeclareLaunchArgument('sim',
                              default_value='false',
                              description='Use simulation instead of real hardware'),
        OpaqueFunction(function=_launch_setup),
    ])
