"""
capture_topdown.py

One-shot capture of the birdseye_camera's view (see the birdseye_camera
model in worlds/vollebekk_20250624.sdf) for the row-marking step
(mark_rows.py). Saves a PNG plus a YAML sidecar recording the camera's
known static pose and intrinsics, needed to convert pixel clicks on that
PNG back into world (x, y) coordinates.

The camera's world pose here MUST match the birdseye_camera <pose> in the
world SDF -- it's not read from TF (there's no robot_state_publisher for a
bare static world model), just duplicated as parameters with the same
defaults. If the SDF pose ever changes, update the defaults below too.
"""

import os

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CaptureTopdown(Node):

    def __init__(self):
        super().__init__('capture_topdown')

        self.declare_parameter(
            'output_dir',
            '/run/media/thiane/59B6290966A08007/projects/nmbu/thorvald_ros2_ws/captures')
        # Filename tag for topdown_<tag>.png/.yaml -- must be set by the
        # launch file (defaults to '' here only so a bare `ros2 run` gives
        # a clear param-missing style error instead of silently reusing
        # '20250624' and overwriting another date's capture).
        self.declare_parameter('output_tag', '')
        # Must match whatever pose capture_topdown.launch.py actually spawned
        # the birdseye camera at for this run (cam_x/y/z/yaw launch args).
        self.declare_parameter('camera_x', -36.4446)
        self.declare_parameter('camera_y', 1.7231)
        self.declare_parameter('camera_z', 39.0)
        self.declare_parameter('camera_roll', 0.0)
        self.declare_parameter('camera_pitch', 1.5708)
        self.declare_parameter('camera_yaw', 0.51245)
        # Assumed flat ground plane for pixel->world projection: mean of the
        # mesh's z range (-2.116 to 0.421 for the 2025-06-24 survey).
        self.declare_parameter('ground_plane_z', -0.8475)

        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._output_tag = self.get_parameter('output_tag').get_parameter_value().string_value
        if not self._output_tag:
            self.get_logger().fatal('Parameter output_tag is required but not set')
            raise RuntimeError('output_tag parameter missing')
        os.makedirs(self._output_dir, exist_ok=True)

        self._bridge = CvBridge()
        self._image_msg = None
        self._camera_info_msg = None

        self.create_subscription(Image, '/birdseye_camera', self._image_cb, 10)
        self.create_subscription(CameraInfo, '/birdseye_camera/camera_info', self._info_cb, 10)

        self.get_logger().info('Waiting for one /birdseye_camera image + camera_info...')

    def _image_cb(self, msg):
        if self._image_msg is None:
            self._image_msg = msg
            self._maybe_finish()

    def _info_cb(self, msg):
        if self._camera_info_msg is None:
            self._camera_info_msg = msg
            self._maybe_finish()

    def _maybe_finish(self):
        if self._image_msg is None or self._camera_info_msg is None:
            return

        image_path = os.path.join(self._output_dir, f'topdown_{self._output_tag}.png')
        meta_path = os.path.join(self._output_dir, f'topdown_{self._output_tag}.yaml')

        cv_image = self._bridge.imgmsg_to_cv2(self._image_msg, desired_encoding='bgr8')
        cv2.imwrite(image_path, cv_image)

        info = self._camera_info_msg
        meta = {
            'image_path': image_path,
            'width': int(info.width),
            'height': int(info.height),
            'intrinsics': {
                'fx': float(info.k[0]),
                'fy': float(info.k[4]),
                'cx': float(info.k[2]),
                'cy': float(info.k[5]),
            },
            'camera_pose': {
                'x': self.get_parameter('camera_x').get_parameter_value().double_value,
                'y': self.get_parameter('camera_y').get_parameter_value().double_value,
                'z': self.get_parameter('camera_z').get_parameter_value().double_value,
                'roll': self.get_parameter('camera_roll').get_parameter_value().double_value,
                'pitch': self.get_parameter('camera_pitch').get_parameter_value().double_value,
                'yaw': self.get_parameter('camera_yaw').get_parameter_value().double_value,
            },
            'ground_plane_z': self.get_parameter('ground_plane_z').get_parameter_value().double_value,
        }
        with open(meta_path, 'w') as f:
            yaml.safe_dump(meta, f, default_flow_style=False)

        self.get_logger().info(f'Saved {image_path} and {meta_path}')
        self.get_logger().info('Done -- shutting down.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CaptureTopdown()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
