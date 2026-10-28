"""
label_row_center.py

Sanity check for the "annotation-free" label mechanism itself, before
trusting it for real dataset generation: projects each marked row's
centerline (world x, y from rows_file) into the live front_camera image
using the robot's own /odom pose plus the camera's known, fixed mount
offset (camera_height/camera_pitch/camera_x_offset -- must match whatever
farm_sim.launch.py was given, see gz_camera.xacro), and saves annotated
frames so you can eyeball whether the projected line actually tracks the
visible soil path in the recorded video.

This draws the *whole* projected polyline (not a single row-center pixel
column) so misalignment is easy to spot by eye -- it's a visual check,
not the final per-frame label extractor.

Math is the forward-projection mirror of mark_rows.pixel_to_world's
ray-cast: world point -> camera-local sensor frame (via the world pose
composed from /odom's full orientation quaternion, not just yaw, since
the robot can pitch/roll slightly) -> optical frame -> pinhole projection
using /front_camera/camera_info's live intrinsics.

Usage (with farm_sim.launch.py already running, follow:=true, against
the SAME rows_file):
  ros2 run farmsim_perception label_row_center --ros-args \
      -p rows_file:=/path/to/rows_centerline.yaml \
      -p output_dir:=/path/to/label_check
"""

import math
import os

import cv2
import message_filters
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from farmsim_farm.row_geometry import load_rows


def _quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _pitch_matrix(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _resample_polyline(points, spacing):
    pts = np.array(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total < 1e-6:
        return [pts[0]]
    n = max(2, int(total // spacing) + 1)
    targets = np.linspace(0.0, total, n)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = []
    for t in targets:
        idx = min(max(np.searchsorted(cum, t) - 1, 0), len(seg_len) - 1)
        local_t = (t - cum[idx]) / seg_len[idx] if seg_len[idx] > 1e-9 else 0.0
        out.append(pts[idx] + local_t * seg[idx])
    return out


class LabelRowCenter(Node):

    def __init__(self):
        super().__init__('label_row_center')
        self.declare_parameter('rows_file', '')
        self.declare_parameter('camera_height', 0.90)
        self.declare_parameter('camera_pitch', 0.15)
        self.declare_parameter('camera_x_offset', 0.80)
        self.declare_parameter('ground_plane_z', -0.8475)
        # See live_label_teleop.py's identical parameter for why this
        # matters: rows_file's (x,y) points were generated against a
        # SPECIFIC map's ground_plane_z (mark_rows.pixel_to_world /
        # segment_rows.py); reusing a different map's value here (the
        # default above is 20250624's) reintroduces the flat-plane error
        # the original approximation was meant to bound, inconsistently
        # between the two directions.
        self.declare_parameter('topdown_meta_file', '')
        self.declare_parameter('output_dir', '/tmp/label_check')
        self.declare_parameter('save_every_n', 15)
        self.declare_parameter('resample_spacing', 2.0)

        rows_file = self.get_parameter('rows_file').get_parameter_value().string_value
        if not rows_file:
            self.get_logger().fatal('Parameter rows_file is required but not set')
            raise RuntimeError('rows_file parameter missing')

        spacing = self.get_parameter('resample_spacing').get_parameter_value().double_value
        # Marked rows are often just 2-3 sparse points (e.g. a straight
        # lane's start/end, tens of meters apart). Drawing that directly
        # means whichever point isn't near the horizon is behind the
        # camera -- nothing useful in the visible foreground to check
        # alignment against. Resample densely so some points always fall
        # in view regardless of where the robot currently is on the row.
        self._rows = [_resample_polyline(row, spacing) for row in load_rows(rows_file)]
        if not self._rows:
            self.get_logger().fatal(f'No usable rows loaded from {rows_file}')
            raise RuntimeError('rows_file has no usable rows')

        self._ground_z = self.get_parameter('ground_plane_z').get_parameter_value().double_value
        meta_file = self.get_parameter('topdown_meta_file').get_parameter_value().string_value
        if meta_file:
            with open(meta_file, 'r') as f:
                meta = yaml.safe_load(f)
            self._ground_z = float(meta['ground_plane_z'])
            self.get_logger().info(
                f'ground_plane_z overridden from {meta_file}: {self._ground_z}')
        self._cam_h = self.get_parameter('camera_height').get_parameter_value().double_value
        self._cam_pitch = self.get_parameter('camera_pitch').get_parameter_value().double_value
        self._cam_x = self.get_parameter('camera_x_offset').get_parameter_value().double_value
        self._save_every_n = self.get_parameter('save_every_n').get_parameter_value().integer_value
        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        os.makedirs(self._output_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._intrinsics = None
        self._frame_count = 0
        self._saved_count = 0

        self.create_subscription(CameraInfo, '/front_camera/camera_info', self._info_cb, 10)

        # /odom (20Hz) and /front_camera (15Hz) are two independent, freely
        # running publishers -- grabbing "whatever /odom message arrived
        # last" (the previous approach) can be up to one odom period stale
        # relative to the image's own capture time (confirmed empirically
        # from a recorded bag: ~25ms average, ~49ms worst-case skew, out of
        # a 66ms camera frame period). That's invisible while the robot's
        # pose is changing slowly, but during a bump/bounce -- exactly when
        # pitch/roll is changing fastest -- it's enough for the projected
        # row to visibly jitter relative to what the image actually shows.
        # Pairing each image with the odom message closest to its own
        # header stamp (within slop) removes that lag.
        odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        image_sub = message_filters.Subscriber(self, Image, '/front_camera')
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [odom_sub, image_sub], queue_size=30, slop=0.06)
        self._sync.registerCallback(self._synced_cb)

        self.get_logger().info(
            f'Loaded {len(self._rows)} row(s) from {rows_file}; '
            f'saving annotated frames to {self._output_dir}')

    def _info_cb(self, msg):
        if self._intrinsics is None:
            self._intrinsics = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _synced_cb(self, odom_msg, msg):
        self._frame_count += 1
        if self._intrinsics is None:
            return
        if self._frame_count % self._save_every_n != 0:
            return

        fx, fy, cx, cy = self._intrinsics
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        r_wb = _quat_to_matrix(ori.x, ori.y, ori.z, ori.w)
        r_bc = _pitch_matrix(self._cam_pitch)
        r_wc = r_wb @ r_bc
        t_wb = np.array([pos.x, pos.y, pos.z])
        t_bc = np.array([self._cam_x, 0.0, self._cam_h])
        t_wc = t_wb + r_wb @ t_bc

        canvas = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8').copy()
        img_h, img_w = canvas.shape[:2]

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                  (255, 0, 255), (255, 165, 0), (128, 0, 255)]

        for row_idx, points in enumerate(self._rows):
            color = colors[row_idx % len(colors)]
            pixels = []
            for p in points:
                p_w = np.array([p[0], p[1], self._ground_z])
                d_w = p_w - t_wc
                d_sensor = r_wc.T @ d_w
                # Sensor (REP-103: X fwd/Y left/Z up) -> optical (Z fwd/X right/Y down).
                x_opt, y_opt, z_opt = -d_sensor[1], -d_sensor[2], d_sensor[0]
                if z_opt <= 0.05:
                    pixels.append(None)
                    continue
                u = fx * x_opt / z_opt + cx
                v = fy * y_opt / z_opt + cy
                pixels.append((int(round(u)), int(round(v))))

            for a, b in zip(pixels, pixels[1:]):
                if a is not None and b is not None:
                    cv2.line(canvas, a, b, color, 3)
            for p in pixels:
                if p is not None and 0 <= p[0] < img_w and 0 <= p[1] < img_h:
                    cv2.circle(canvas, p, 4, color, -1)

        out_path = os.path.join(self._output_dir, f'frame_{self._saved_count:04d}.png')
        cv2.imwrite(out_path, canvas)
        self._saved_count += 1
        self.get_logger().info(f'Saved {out_path}')


def main(args=None):
    rclpy.init(args=args)
    node = LabelRowCenter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
