from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    pkg = get_package_share_directory('thorvald_twist_mux')

    twist_mux_node = Node(
        package='thorvald_twist_mux',
        executable='twist_mux_node',
        name='twist_mux',
        parameters=[os.path.join(pkg, 'config', 'twist_mux_topics.yaml')],
        remappings=[('cmd_vel', 'platform/cmd_vel')],
        output='screen',
    )

    return LaunchDescription([twist_mux_node])
