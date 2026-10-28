"""
record_camera_poses.py

Phase 4 of the mesh-based label-rendering pipeline: record the front
camera's RGB frames together with the robot's GROUND-TRUTH world pose per
frame -- not /odom. /odom comes from gz-sim's OdometryPublisher system,
which (per Thorvald's own odometry docstrings) is meant to model a real
robot's wheel-odometry-style estimate; even though it's numerically exact
in this sim (no noise injected), there's no reason to route the label
pipeline through anything with "odometry" semantics when gz-sim already
publishes true world pose directly. Subscribes to gz-transport's
/world/<world>/dynamic_pose/info (gz.msgs.Pose_V) natively instead --
this is the same scene-broadcaster stream `gz topic echo` shows a
"thorvald" entry in with a plain top-level position and orientation.

Why not bridge this through ros_gz_bridge like every other topic in this
package: ros_gz_bridge's Pose_V -> tf2_msgs/TFMessage conversion drops the
per-entry `name` field (confirmed empirically -- frame_id/child_frame_id
come through empty), leaving no reliable way to pick "thorvald" out of
the list of per-link poses it's bundled with. Subscribing directly via
the gz.transport Python bindings keeps the name.

Per frame, writes:
  <output_dir>/frames/frame_%06d.png
  <output_dir>/poses.jsonl  -- one JSON object per frame:
    {"frame": "frame_000123.png", "stamp": <image header stamp, sec.nanosec>,
     "T_world_optical": <4x4 row-major>, "pose_age_s": <|pose_stamp - image_stamp|>,
     "cmd_vel": {"linear_x": ..., "angular_z": ...} or null, "cmd_vel_age_s": ... or null,
     "joint_states": {"name": [...], "position": [...], "velocity": [...]} or null,
     "joint_states_age_s": <|joint_state_stamp - image_stamp|> or null}
  cmd_vel/joint_states are supplementary robot-state logging (the commanded
  action and the actual per-wheel steering/velocity state) for training
  uses beyond the label-rendering pipeline itself -- NOT hard-gated on
  staleness the way pose is (pose feeds camera projection math directly;
  these are just informational). cmd_vel has no header/stamp field
  (geometry_msgs/Twist), so its age is measured via this node's own clock
  at message-receipt time rather than against the image's header stamp
  the way pose_age_s/joint_states_age_s are -- an internally-consistent
  delta, not a cross-clock comparison, and less precise as a result;
  fine for "was this stale" sanity checks, not for anything exact.
Once, at start:
  <output_dir>/camera_info.json -- fx, fy, cx, cy, width, height, distortion

T_world_optical composes: world_from_base_link (gz-transport ground truth)
-> base_link_from_camera_link (gz_camera.xacro's fixed joint: xyz=(camera_x_offset,
0,camera_height), rpy=(0,camera_pitch,0)) -> camera_link_from_optical (fixed
REP-103 sensor -> optical-frame rotation, same convention used everywhere
else in this package, e.g. live_label_teleop.py).

Usage (with farm_sim.launch.py already running):
  ros2 run farmsim_perception record_camera_poses --ros-args \
      -p world_name:=vollebekk_farm_20250425 \
      -p output_dir:=/path/to/recordings/label_run_20250425 \
      -p camera_height:=0.90 -p camera_pitch:=0.15 -p camera_x_offset:=0.80 \
      -p save_every_n:=5
"""

import json
import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gz.msgs.pose_v_pb2 import Pose_V
from gz.transport import Node as GzNode
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState

from farmsim_farm.camera_geometry import R_SENSOR_TO_OPTICAL, rpy_matrix


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pitch_matrix(pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class RecordCameraPoses(Node):

    def __init__(self):
        super().__init__('record_camera_poses')
        self.declare_parameter('world_name', 'vollebekk_farm_20250425')
        self.declare_parameter('output_dir', '/tmp/label_run')
        self.declare_parameter('camera_height', 0.90)
        self.declare_parameter('camera_pitch', 0.15)
        self.declare_parameter('camera_x_offset', 0.80)
        self.declare_parameter('camera_roll', 0.0)
        self.declare_parameter('camera_yaw', 0.0)
        self.declare_parameter('save_every_n', 5)
        self.declare_parameter('robot_model_name', 'thorvald')
        # Ground-truth pose messages and camera frames are two independent
        # streams (gz-transport pose vs ROS image) -- same class of skew
        # risk as /odom+image before, mitigated the same way: pair each
        # saved frame with the closest-in-time pose sample rather than
        # "whatever arrived last," and record the actual age so a bad
        # pairing is visible in the data rather than silently trusted.
        self.declare_parameter('max_pose_age_s', 0.1)

        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._frames_dir = os.path.join(self._output_dir, 'frames')
        os.makedirs(self._frames_dir, exist_ok=True)

        self._cam_h = self.get_parameter('camera_height').get_parameter_value().double_value
        self._cam_pitch = self.get_parameter('camera_pitch').get_parameter_value().double_value
        self._cam_x = self.get_parameter('camera_x_offset').get_parameter_value().double_value
        self._cam_roll = self.get_parameter('camera_roll').get_parameter_value().double_value
        self._cam_yaw = self.get_parameter('camera_yaw').get_parameter_value().double_value
        self._save_every_n = self.get_parameter('save_every_n').get_parameter_value().integer_value
        self._robot_name = self.get_parameter('robot_model_name').get_parameter_value().string_value
        self._max_pose_age = self.get_parameter('max_pose_age_s').get_parameter_value().double_value

        r_bc = rpy_matrix(self._cam_roll, self._cam_pitch, self._cam_yaw)
        self._t_bc = np.array([self._cam_x, 0.0, self._cam_h])
        self._r_bc = r_bc

        self._bridge = CvBridge()
        self._intrinsics_written = False
        self._frame_count = 0
        self._saved_count = 0
        self._latest_pose = None  # (stamp_sec_float, position xyz, quat xyzw)
        self._latest_cmd_vel = None  # (recv_time (rclpy.time.Time), linear_x, angular_z)
        self._latest_joint_state = None  # (stamp_sec_float, name list, position list, velocity list)

        self._poses_file = open(os.path.join(self._output_dir, 'poses.jsonl'), 'w')

        world_name = self.get_parameter('world_name').get_parameter_value().string_value
        topic = f'/world/{world_name}/dynamic_pose/info'
        self._gz_node = GzNode()
        ok = self._gz_node.subscribe(Pose_V, topic, self._gz_pose_cb)
        if not ok:
            self.get_logger().fatal(f'Failed to subscribe to gz-transport topic {topic}')
            raise RuntimeError(f'gz-transport subscribe failed: {topic}')
        self.get_logger().info(f'Subscribed to ground-truth pose topic {topic}')

        self.create_subscription(CameraInfo, '/front_camera/camera_info', self._info_cb, 10)
        self.create_subscription(Image, '/front_camera', self._image_cb, 10)
        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        self.get_logger().info(
            f'Recording to {self._output_dir} (frames every {self._save_every_n} '
            f'front_camera messages)')

    def _gz_pose_cb(self, msg: Pose_V):
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        for pose in msg.pose:
            if pose.name == self._robot_name:
                p = pose.position
                o = pose.orientation
                self._latest_pose = (
                    stamp,
                    np.array([p.x, p.y, p.z]),
                    np.array([o.x, o.y, o.z, o.w]),
                )
                return

    def _cmd_vel_cb(self, msg: Twist):
        self._latest_cmd_vel = (self.get_clock().now(), float(msg.linear.x), float(msg.angular.z))

    def _joint_state_cb(self, msg: JointState):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._latest_joint_state = (stamp, list(msg.name), list(msg.position), list(msg.velocity))

    def _info_cb(self, msg):
        if self._intrinsics_written:
            return
        info = {
            'fx': float(msg.k[0]), 'fy': float(msg.k[4]),
            'cx': float(msg.k[2]), 'cy': float(msg.k[5]),
            'width': int(msg.width), 'height': int(msg.height),
            'distortion_model': msg.distortion_model,
            'distortion': list(msg.d),
        }
        with open(os.path.join(self._output_dir, 'camera_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        self._intrinsics_written = True
        self.get_logger().info(f'Wrote camera_info.json: {info}')

    def _image_cb(self, msg):
        self._frame_count += 1
        if self._frame_count % self._save_every_n != 0:
            return
        if self._latest_pose is None:
            self.get_logger().warning('No ground-truth pose received yet, skipping frame')
            return

        image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose_stamp, t_wb, quat = self._latest_pose
        age = abs(image_stamp - pose_stamp)
        if age > self._max_pose_age:
            self.get_logger().warning(
                f'Pose is {age:.3f}s old (> max_pose_age_s={self._max_pose_age}), '
                'skipping frame rather than pairing stale data')
            return

        r_wb = quat_to_matrix(*quat)
        # R_SENSOR_TO_OPTICAL is defined (and used everywhere else in this
        # package, e.g. label_row_center.py) as "sensor-frame vector
        # components -> optical-frame vector components": v_optical =
        # R_SENSOR_TO_OPTICAL @ v_sensor. Composing a WORLD-frame rotation
        # chain needs the opposite direction at this last link --
        # R_camera_link_from_optical, i.e. v_sensor = R @ v_optical -- which
        # is R_SENSOR_TO_OPTICAL's TRANSPOSE (its inverse, since it's
        # orthogonal), not itself. Using it un-transposed here silently
        # produced a badly wrong T_world_optical for every recorded frame
        # (empirically: optical "up" came out ~horizontal in world frame
        # instead of ~vertical) -- caught by check_alignment.py's Part B
        # against real captured frames, since Part A (which only checks
        # render_labels.py's own math against itself) can't see a bug this
        # far upstream of either projection function.
        r_wo = r_wb @ self._r_bc @ R_SENSOR_TO_OPTICAL.T
        t_wc = t_wb + r_wb @ self._t_bc

        t_world_optical = np.eye(4)
        t_world_optical[:3, :3] = r_wo
        t_world_optical[:3, 3] = t_wc

        frame_name = f'frame_{self._saved_count:06d}.png'
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(os.path.join(self._frames_dir, frame_name), cv_image)

        cmd_vel_dict, cmd_vel_age = None, None
        if self._latest_cmd_vel is not None:
            recv_time, lin_x, ang_z = self._latest_cmd_vel
            cmd_vel_dict = {'linear_x': lin_x, 'angular_z': ang_z}
            cmd_vel_age = (self.get_clock().now() - recv_time).nanoseconds * 1e-9

        joint_states_dict, joint_states_age = None, None
        if self._latest_joint_state is not None:
            js_stamp, names, positions, velocities = self._latest_joint_state
            joint_states_dict = {'name': names, 'position': positions, 'velocity': velocities}
            joint_states_age = abs(image_stamp - js_stamp)

        record = {
            'frame': frame_name,
            'stamp': image_stamp,
            'T_world_optical': t_world_optical.tolist(),
            'pose_age_s': age,
            'cmd_vel': cmd_vel_dict,
            'cmd_vel_age_s': cmd_vel_age,
            'joint_states': joint_states_dict,
            'joint_states_age_s': joint_states_age,
        }
        self._poses_file.write(json.dumps(record) + '\n')
        self._poses_file.flush()
        self._saved_count += 1
        if self._saved_count % 10 == 0:
            self.get_logger().info(f'Saved {self._saved_count} frames (pose age {age*1000:.1f}ms)')

    def destroy_node(self):
        self._poses_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecordCameraPoses()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
