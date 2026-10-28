"""
joint_command_splitter.py

Bridges thorvald_base's twist_to_joints node (the real Thorvald 4WS/4WD
kinematics, unmodified) to the per-joint gz-sim plugins added to
thorvald_model.xacro under control_backend='gz_sim'.

twist_to_joints publishes one Float64MultiArray per group:
  /wheel_velocity_controller/commands     order: wheel0..wheelN
  /steering_position_controller/commands  order: steering0..steeringN

gz-sim's JointController / JointPositionController plugins each subscribe
to their own per-joint std_msgs/Float64 topic (wheel{i}/cmd_vel,
steering{i}/cmd_pos). This node splits the arrays into those per-joint
topics.

Parameters:
  robot_config  (string)  Absolute path to the robot YAML config file, used
                          only to determine the number of drives (same
                          config passed to twist_to_joints).
"""

import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray


class JointCommandSplitter(Node):

    def __init__(self):
        super().__init__('joint_command_splitter')

        self.declare_parameter('robot_config', '')
        config_path = self.get_parameter('robot_config').get_parameter_value().string_value
        if not config_path:
            self.get_logger().fatal('Parameter robot_config is required but not set')
            raise RuntimeError('robot_config parameter missing')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        motor_drives = config.get('motor_drives', {})
        n = 0
        while f'drive{n}' in motor_drives:
            n += 1
        if n == 0:
            self.get_logger().fatal('No motor drives found in config: ' + config_path)
            raise RuntimeError('No motor drives in config')
        self._n = n
        self.get_logger().info(f'Splitting commands for {n} drives')

        self._wheel_pubs = [
            self.create_publisher(Float64, f'wheel{i}/cmd_vel', 10) for i in range(n)
        ]
        self._steer_pubs = [
            self.create_publisher(Float64, f'steering{i}/cmd_pos', 10) for i in range(n)
        ]

        self.create_subscription(
            Float64MultiArray, '/wheel_velocity_controller/commands', self._wheel_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/steering_position_controller/commands', self._steer_cb, 10)

    def _wheel_cb(self, msg: Float64MultiArray):
        for i, value in enumerate(msg.data):
            if i >= self._n:
                break
            self._wheel_pubs[i].publish(Float64(data=value))

    def _steer_cb(self, msg: Float64MultiArray):
        for i, value in enumerate(msg.data):
            if i >= self._n:
                break
            self._steer_pubs[i].publish(Float64(data=value))


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandSplitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
