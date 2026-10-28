"""
node.py  (ros2 run farmsim_control controller_node)

Drives the robot along a configured FieldPath (cfg/paths/*.yaml).

  in-row   the configured strategy (laws.py / fuzzy.py) on the row state from
           `pose_source`: 'gt' = ground-truth /odom against the path's row
           line, 'perception' = the CNN's estimate on /perception/row_state
  headland pure pursuit on ground truth between passes (vision-based headland
           turning is out of scope)

Progress along the path, and so the switch between row and headland, always
uses ground truth. Every cycle is logged to `log_file` (JSON lines) with the
ground-truth error, the error the controller actually used and, whenever
perception_node is running (in either mode), the CNN estimate, so
run_control.py / run_perception.py --live can score both against the truth.

Parameters:
  config_file  YAML with a `control:` block (cfg/control.yaml)
  path_file    YAML path (cfg/paths/*.yaml)
  pose_source  'gt' | 'perception'
  log_file     JSON-lines output, '' disables
  perception_timeout_s  stop (v=0) when the CNN estimate is older than this
"""

import json
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from farmsim_control.laws import ControlState, make_controller
from farmsim_control.runtime import CommandFilter, headland_command
from farmsim_farm.paths import FieldPath, PathTracker
from farmsim_utils import row_state


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControllerNode(Node):

    def __init__(self):
        super().__init__('controller_node')
        self.declare_parameter('config_file', '')
        self.declare_parameter('path_file', '')
        self.declare_parameter('pose_source', 'gt')
        self.declare_parameter('log_file', '')
        self.declare_parameter('perception_timeout_s', 0.5)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        with open(p('config_file')) as f:
            cfg = yaml.safe_load(f)['control']
        self.cfg = cfg
        self.pose_source = p('pose_source')
        if self.pose_source not in ('gt', 'perception'):
            raise RuntimeError(f"pose_source must be 'gt' or 'perception', got {self.pose_source!r}")
        self.controller = make_controller(cfg).eval()
        self.path = FieldPath.load(p('path_file'))
        self.tracker = PathTracker(self.path)
        rt = cfg.get('runtime', {})
        self.filter = CommandFilter(cfg.get('max_angular', 0.5), rt.get('max_angular_rate', 0.6),
                                    rt.get('max_linear_accel', 0.5), rt.get('perturb_std', 0.0),
                                    rt.get('seed'))
        self.headland = cfg.get('headland', {'speed': 0.25, 'lookahead_m': 1.2})
        self.timeout = p('perception_timeout_s')
        self.perception = None
        self.perception_time = 0.0
        self.last_t = None
        self.done = False
        self.log = open(p('log_file'), 'w') if p('log_file') else None

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Float32MultiArray, row_state.TOPIC, self.on_perception, 10)
        self.get_logger().info(
            f"strategy={cfg.get('strategy')} pose_source={self.pose_source} "
            f"passes={[vars(x) for x in self.path.passes]} length={self.tracker.total:.1f} m")

    def on_perception(self, msg):
        self.perception = row_state.decode(msg)
        self.perception_time = time.monotonic()

    def on_odom(self, msg):
        if self.done:
            self.pub.publish(Twist())
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self.last_t is None else now - self.last_t
        self.last_t = now
        pos = msg.pose.pose.position
        yaw = _yaw(msg.pose.pose.orientation)
        r = msg.twist.twist.angular.z
        speed = msg.twist.twist.linear.x
        ref = self.tracker.update(pos.x, pos.y, yaw)

        used = {'lateral_offset': ref['lateral_offset'], 'heading_error': ref['heading_error'], 'confidence': 1.0}
        if ref['done']:
            self.done = True
            v, w = 0.0, 0.0
            self.get_logger().info('path complete')
        elif ref['kind'] == 'row':
            if self.pose_source == 'perception':
                if self.perception is None or time.monotonic() - self.perception_time > self.timeout:
                    used = None
                else:
                    used = {k: self.perception[k] for k in ('lateral_offset', 'heading_error', 'confidence')}
            if used is None:
                v, w = 0.0, 0.0
            else:
                v, w = self.controller.step(ControlState(
                    used['lateral_offset'], used['heading_error'], r, speed,
                    ref['dist_to_end'], used['confidence']))
        else:
            target = self.tracker.lookahead_point(self.headland['lookahead_m'])
            v, w = headland_command(pos.x, pos.y, yaw, target, self.headland['speed'],
                                    self.headland['lookahead_m'], self.controller.max_angular)
        v, w = self.filter(v, w, dt) if not self.done else (0.0, 0.0)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.pub.publish(cmd)
        if self.log:
            est = self.perception if self.perception and time.monotonic() - self.perception_time <= self.timeout else None
            self.log.write(json.dumps({
                't': now, 'x': pos.x, 'y': pos.y, 'yaw': yaw, 'kind': ref['kind'], 'row': ref['row'],
                'pass': ref['pass_index'], 's': ref['s'], 'e_gt': ref['lateral_offset'],
                'psi_gt': ref['heading_error'], 'e_used': None if used is None else used['lateral_offset'],
                'psi_used': None if used is None else used['heading_error'],
                'conf': None if used is None else used['confidence'],
                'e_est': est and est['lateral_offset'], 'psi_est': est and est['heading_error'],
                'conf_est': est and est['confidence'],
                'v': v, 'w': w, 'done': self.done, 'total': self.tracker.total}) + '\n')
            self.log.flush()


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(ControllerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
