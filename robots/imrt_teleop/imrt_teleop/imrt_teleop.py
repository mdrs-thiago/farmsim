#!/usr/bin/env python3
"""imrt_teleop.py

ROS2 node that converts sensor_msgs/msg/Joy messages (from a joystick/gamepad)
into geometry_msgs/msg/Twist velocity commands.

Mapping for the provided NMBU virtual gamepad (see the supplied gamepad.py):
  axes[0] : left stick up/down   (up is +1.0)
  axes[1] : left stick left/right
  axes[2] : right stick up/down
  axes[3] : right stick left/right (right is +1.0)

Assignment requirements:
  - Subscribe to Joy on the joy topic
  - Publish Twist to cmd_vel whenever a Joy message is received
  - Left stick up/down -> linear.x (up = forward)
  - Right stick side/side -> angular.z (right = clockwise)
  - max linear speed  = +/- 0.7 m/s
  - max angular speed = +/- 0.9 rad/s
"""

from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


@dataclass
class TeleopConfig:
    """Configuration for the Joy->Twist mapping."""

    left_stick_ud_axis: int = 0       # axes[0]
    right_stick_lr_axis: int = 3      # axes[3]
    max_linear_x: float = 0.7
    max_angular_z: float = 0.9
    angular_clockwise_when_right: bool = True


class ImrtTeleop(Node):
    """Teleoperation node."""

    def __init__(self) -> None:
        super().__init__("imrt_teleop")

        # Parameters make the node easy to reuse with different controllers.
        self.declare_parameter("left_stick_ud_axis", 0)
        self.declare_parameter("right_stick_lr_axis", 3)
        self.declare_parameter("max_linear_x", 0.7)
        self.declare_parameter("max_angular_z", 0.9)

        self._cfg = TeleopConfig(
            left_stick_ud_axis=int(self.get_parameter("left_stick_ud_axis").value),
            right_stick_lr_axis=int(self.get_parameter("right_stick_lr_axis").value),
            max_linear_x=float(self.get_parameter("max_linear_x").value),
            max_angular_z=float(self.get_parameter("max_angular_z").value),
        )

        self._twist_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._joy_sub = self.create_subscription(Joy, "/joy", self._joy_callback, 10)

        self.get_logger().info(
            "imrt_teleop started. Subscribing to /joy, publishing cmd_vel."
        )

    def _joy_callback(self, msg: Joy) -> None:
        """Convert incoming Joy message to a Twist command."""
        if len(msg.axes) <= max(self._cfg.left_stick_ud_axis, self._cfg.right_stick_lr_axis):
            self.get_logger().warning(
                "Joy message has too few axes (%d). Expected at least %d."
                % (len(msg.axes), max(self._cfg.left_stick_ud_axis, self._cfg.right_stick_lr_axis) + 1)
            )
            return

        left_ud = float(msg.axes[self._cfg.left_stick_ud_axis])
        right_lr = float(msg.axes[self._cfg.right_stick_lr_axis])

        twist = Twist()
        twist.linear.x = self._cfg.max_linear_x * left_ud

        # ROS convention: +angular.z is counter-clockwise.
        # Requirement: right stick to the right should be clockwise => negative angular.z.
        twist.angular.z = -self._cfg.max_angular_z * right_lr

        self._twist_pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImrtTeleop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
