"""
thorvald_teleop.launch.py

Starts the NMBU virtual joystick and the Joy-to-Twist converter for Thorvald.

Usage:
  ros2 launch thorvald_sim thorvald_teleop.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    virtual_gamepad = Node(
        package="imrt_virtual_joy",
        executable="gamepad_talker",
        name="gamepad_talker",
        output="screen",
    )

    joy_to_cmd_vel = Node(
        package="imrt_teleop",
        executable="imrt_teleop",
        name="imrt_teleop",
        output="screen",
        parameters=[{
            "max_linear_x": 0.7,
            "max_angular_z": 0.9,
        }],
    )

    return LaunchDescription([
        virtual_gamepad,
        joy_to_cmd_vel,
    ])
