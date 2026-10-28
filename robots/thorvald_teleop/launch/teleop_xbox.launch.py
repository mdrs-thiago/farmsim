from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    pkg = get_package_share_directory('thorvald_teleop')

    joy_dev_arg = DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0')
    autorepeat_arg = DeclareLaunchArgument('joy_autorepeat_rate', default_value='20.0')
    deadzone_arg = DeclareLaunchArgument('joy_deadzone', default_value='0.05')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': LaunchConfiguration('joy_dev'),
            'autorepeat_rate': LaunchConfiguration('joy_autorepeat_rate'),
            'deadzone': LaunchConfiguration('joy_deadzone'),
        }],
        output='screen',
    )

    teleop_node = Node(
        package='thorvald_teleop',
        executable='teleop_node',
        name='teleop_node',
        parameters=[os.path.join(pkg, 'config', 'xbox.yaml')],
        remappings=[('cmd_vel', 'teleop_joy/cmd_vel'),
                    ('joy_priority', 'teleop_joy/joy_priority')],
        output='screen',
    )

    return LaunchDescription([
        joy_dev_arg, autorepeat_arg, deadzone_arg,
        joy_node, teleop_node,
    ])
