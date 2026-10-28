#!/usr/bin/env python3
"""gamepad_talker.py

ROS2 node that runs the provided PyQt5 virtual gamepad and publishes its state as
sensor_msgs/msg/Joy messages on the /joy topic.

This file is intended to be used as a ROS2 console_script entry point:
  ros2 run imrt_virtual_joy gamepad_talker
"""

import threading
from typing import List, Tuple

from PyQt5 import QtWidgets

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

try:
    # When executed as part of a Python package.
    from .gamepad import GamePad
except ImportError:
    # When executed directly during development.
    from gamepad import GamePad


class GamepadTalker(Node):
    """Publishes Joy messages based on the GUI state."""

    def __init__(self, gamepad: GamePad):
        super().__init__("gamepad_talker")

        # Publish to an absolute topic name so the topic remains /joy even if the node
        # is launched in a namespace (as in the provided launch snippet).
        self.publisher_ = self.create_publisher(Joy, "/joy", 10)

        # The gamepad GUI instance (lives in the Qt main thread).
        self._gamepad = gamepad

        # Publish at 20 Hz (0.05 s), matching the assignment starter.
        self._timer = self.create_timer(0.05, self._timer_callback)

        self.get_logger().info("Virtual gamepad ROS2 talker started. Publishing /joy at 20 Hz.")

    def _timer_callback(self) -> None:
        """Read latest state from the GamePad and publish a Joy message."""
        axes, buttons = self._get_gamepad_state()

        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Ensure types match the Joy message definition:
        #  - axes: list[float32]
        #  - buttons: list[int32]
        msg.axes = [float(a) for a in axes]
        msg.buttons = [int(b) for b in buttons]

        self.publisher_.publish(msg)

    def _get_gamepad_state(self) -> Tuple[List[float], List[bool]]:
        """Get (axes, buttons) from the GUI.

        The provided GamePad returns:
          axes   : list[float] in range [-1.0, 1.0]
          buttons: list[bool]  (pressed / not pressed)
        """
        axes, buttons = self._gamepad._loop_event()
        return axes, buttons


def _start_ros_thread(gamepad: GamePad) -> None:
    """Entry point for the ROS spinning thread."""
    rclpy.init()
    node = GamepadTalker(gamepad)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    """Start Qt GUI in the main thread and spin ROS2 in a background thread."""
    app = QtWidgets.QApplication([])

    # Optional: keep the same styling as the starter code.
    app.setStyleSheet(
        """
        QMainWindow{background-color: #ECEFF4;
                    border: 8px double #4C566A;
                    border-radius: 20px;}
        QPushButton{background-color: #88C0D0;
                    color: #4C566A}
        """
    )

    gamepad = GamePad(autorepeat=False)

    ros_thread = threading.Thread(target=_start_ros_thread, args=(gamepad,), daemon=True)
    ros_thread.start()

    # Run the GUI until the window is closed.
    app.exec_()

    # ROS2 node will shut down when the process exits. Join for clean shutdown.
    ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
