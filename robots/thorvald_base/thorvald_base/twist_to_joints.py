"""
twist_to_joints.py

Converts /cmd_vel (geometry_msgs/Twist) to individual wheel velocity and
steering position commands for use with ros2_control
JointGroupVelocityController and JointGroupPositionController.

Kinematics ported from thorvald_base/pltf_clc_std (ROS1).

Topics published:
  /wheel_velocity_controller/commands  [std_msgs/Float64MultiArray]
    Wheel joint angular velocities in rad/s, order: wheel0..wheelN
  /steering_position_controller/commands  [std_msgs/Float64MultiArray]
    Steering joint positions in rad, order: steering0..steeringN

Parameters:
  robot_config  (string)  Absolute path to the robot YAML config file
                          (e.g. thorvald_ii_4wd4ws_std.yaml)
  max_steering_rate  (double)  Maximum steering angle change rate in rad/s.
"""

import math
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class TwistToJoints(Node):

    def __init__(self):
        super().__init__('twist_to_joints')

        self.declare_parameter('robot_config', '')
        self.declare_parameter('max_steering_rate', 0.8)
        config_path = self.get_parameter('robot_config').get_parameter_value().string_value
        self._max_steering_rate = (
            self.get_parameter('max_steering_rate').get_parameter_value().double_value
        )

        if not config_path:
            self.get_logger().fatal('Parameter robot_config is required but not set')
            raise RuntimeError('robot_config parameter missing')
        if self._max_steering_rate <= 0.0:
            self.get_logger().fatal('Parameter max_steering_rate must be greater than zero')
            raise RuntimeError('max_steering_rate parameter invalid')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        motor_drives = config.get('motor_drives', {})
        self._drives = []
        i = 0
        while f'drive{i}' in motor_drives:
            d = motor_drives[f'drive{i}']
            self._drives.append({
                'x': float(d['x']),
                'y': float(d['y']),
                'r_wheel': float(d['r_wheel']),
            })
            i += 1

        self._n = len(self._drives)
        if self._n == 0:
            self.get_logger().fatal('No motor drives found in config: ' + config_path)
            raise RuntimeError('No motor drives in config')
        self._last_steers = [0.0] * self._n
        self._last_steer_time = self.get_clock().now()

        self.get_logger().info(
            f'Loaded {self._n} drives from {config_path}')
        self.get_logger().info(
            f'Max steering rate: {self._max_steering_rate:.3f} rad/s')

        self._wheel_pub = self.create_publisher(
            Float64MultiArray, '/wheel_velocity_controller/commands', 10)
        self._steer_pub = self.create_publisher(
            Float64MultiArray, '/steering_position_controller/commands', 10)

        self._cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_cb, 10)

    # ------------------------------------------------------------------
    def _cmd_vel_cb(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        speeds = []
        steers = []

        if wz != 0.0:
            # Turning motion: compute instantaneous centre of rotation
            turn_rad_d = math.sqrt(vx ** 2 + vy ** 2) / wz
            turn_rad_ang = math.atan2(vy, vx)
            turn_rad_x = -turn_rad_d * math.sin(turn_rad_ang)
            turn_rad_y = turn_rad_d * math.cos(turn_rad_ang)

            for d in self._drives:
                dx = turn_rad_x - d['x']
                dy = turn_rad_y - d['y']
                # Steering angle: point each wheel tangent to its ICR circle
                steer = self._normalize_angle(
                    -math.atan2(dx, dy) + math.pi * (wz < 0))
                speed = math.sqrt(dx ** 2 + dy ** 2) * abs(wz)
                steer, speed = self._fold_angle(steer, speed)
                speeds.append(speed / d['r_wheel'])   # m/s → rad/s
                steers.append(steer)
        else:
            # Pure translation
            steer = math.atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else 0.0
            speed_lin = math.sqrt(vx ** 2 + vy ** 2)
            for d in self._drives:
                speeds.append(speed_lin / d['r_wheel'])  # m/s → rad/s
                steers.append(steer)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = speeds
        steer_msg = Float64MultiArray()
        steer_msg.data = self._limit_steering_rate(steers)

        self._wheel_pub.publish(wheel_msg)
        self._steer_pub.publish(steer_msg)

    # ------------------------------------------------------------------
    def _limit_steering_rate(self, steers):
        now = self.get_clock().now()
        dt = (now - self._last_steer_time).nanoseconds * 1e-9

        # Sim time can jump on startup/reset. Keep the limiter stable.
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05

        max_delta = self._max_steering_rate * dt
        limited = []
        for current, target in zip(self._last_steers, steers):
            error = self._normalize_angle(target - current)
            error = max(-max_delta, min(max_delta, error))
            limited.append(self._normalize_angle(current + error))

        self._last_steers = limited
        self._last_steer_time = now
        return limited

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to (-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _fold_angle(steer: float, speed: float):
        """
        If the desired steering angle is outside the joint limit (±π/2),
        flip by π and reverse speed — equivalent motion, within limits.
        """
        if steer > math.pi / 2.0:
            steer -= math.pi
            speed = -speed
        elif steer < -math.pi / 2.0:
            steer += math.pi
            speed = -speed
        return steer, speed


def main(args=None):
    rclpy.init(args=args)
    node = TwistToJoints()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
