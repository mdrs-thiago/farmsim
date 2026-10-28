"""
rl_env.py

Gym-style environment around the simulator for reinforcement learning of
the controller (optimize.py, method: rl). Experimental: it runs, but the
reward and the training loop on top of it are untuned.

  obs     [lateral_offset, heading_error, confidence, yaw_rate, speed, dist_to_end]
          from the CNN (perception_node) and odometry
  action  [v, omega]
  reward  progress along the path (m) - w_offset * |e_gt| * dt - w_heading * |psi_gt| * dt
          - 10 when the ground-truth offset exceeds max_offset_m (episode ends)

Each reset relaunches the sim with the robot at the path's start, so an
episode costs a sim startup (~15 s). The process must run with ROS and the
workspace sourced (rclpy is imported here).
"""

import math
import time

import numpy as np

from farmsim_farm.paths import FieldPath, PathTracker
from farmsim_utils import row_state
from farmsim_utils.pipeline import repo_path, sim_launch_mapping
from farmsim_utils.sim import Sim


class FarmRowEnv:

    def __init__(self, cfg, path_file, checkpoint, log_dir, dt=0.1, max_steps=600, max_offset_m=1.0,
                 w_offset=1.0, w_heading=0.5):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32MultiArray
        self._Twist = Twist
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('farmsim_rl_env')
        self.rclpy = rclpy
        self.pub = self.node.create_publisher(Twist, 'cmd_vel', 10)
        self.node.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.node.create_subscription(Float32MultiArray, row_state.TOPIC, self._on_est, 10)
        self.cfg, self.path_file, self.checkpoint = cfg, repo_path(path_file), repo_path(checkpoint)
        self.log_dir = log_dir
        self.dt, self.max_steps, self.max_offset_m = dt, max_steps, max_offset_m
        self.w_offset, self.w_heading = w_offset, w_heading
        self.sim = None
        self.episode = 0

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                     math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                     msg.twist.twist.angular.z, msg.twist.twist.linear.x)

    def _on_est(self, msg):
        self.est = row_state.decode(msg)

    def _spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.rclpy.spin_once(self.node, timeout_sec=max(0.0, min(0.02, end - time.monotonic())))

    def _obs(self):
        e = self.est or {'lateral_offset': 0.0, 'heading_error': 0.0, 'confidence': 0.0}
        _, _, _, r, v = self.pose
        return np.array([e['lateral_offset'], e['heading_error'], e['confidence'], r, v,
                         self.ref['dist_to_end']], dtype=np.float32)

    def reset(self):
        self.close_sim()
        path = FieldPath.load(self.path_file)
        self.tracker = PathTracker(path)
        x, y, yaw = path.spawn_pose()
        mapping = sim_launch_mapping(self.cfg, {'x': x, 'y': y, 'yaw': yaw}, perception=True,
                                     checkpoint=self.checkpoint)
        self.episode += 1
        self.sim = Sim(mapping, str(self.log_dir / f'sim_ep{self.episode}.log'),
                       startup_wait_s=self.cfg['sim'].get('startup_wait_s', 12)).__enter__()
        self.pose, self.est, self.steps = None, None, 0
        t0 = time.monotonic()
        while (self.pose is None or self.est is None) and time.monotonic() - t0 < 30:
            self._spin(0.2)
        if self.pose is None:
            raise RuntimeError('no /odom after reset; is ROS sourced and the sim up?')
        self.ref = self.tracker.update(*self.pose[:3])
        return self._obs()

    def step(self, action):
        cmd = self._Twist()
        cmd.linear.x, cmd.angular.z = float(action[0]), float(action[1])
        self.pub.publish(cmd)
        s_before = self.ref['s']
        self._spin(self.dt)
        self.ref = self.tracker.update(*self.pose[:3])
        self.steps += 1
        reward = (self.ref['s'] - s_before) - self.dt * (
            self.w_offset * abs(self.ref['lateral_offset']) + self.w_heading * abs(self.ref['heading_error']))
        off_track = self.ref['kind'] == 'row' and abs(self.ref['lateral_offset']) > self.max_offset_m
        if off_track:
            reward -= 10.0
        done = self.ref['done'] or off_track or self.steps >= self.max_steps
        return self._obs(), float(reward), done, {'ref': self.ref, 'off_track': off_track}

    def close_sim(self):
        if self.sim is not None:
            self.pub.publish(self._Twist())
            self.sim.__exit__(None, None, None)
            self.sim = None

    def close(self):
        self.close_sim()
        self.node.destroy_node()
