"""
live_segmentation_view.py

Live visual sanity check, meant to run alongside farm_sim.launch.py
(follow:=true): shows what the crop-row segmentation algorithm
(segment_rows.vegetation_mask) sees on the live front_camera feed, updated
every frame, in its own OpenCV window next to the Gazebo GUI -- so you can
watch segmentation quality in real time as the robot drives instead of only
checking it against the static top-down capture.

Reuses segment_rows.vegetation_mask() as-is: same HSV hue threshold tuned
for this photogrammetry render's foliage vs soil colors. The top-down
capture and the front camera are very different viewing angles/distances,
so don't expect the same crispness as the overlay PNG segment_rows.py
produces -- this is for eyeballing whether foliage is being picked up at
all from ground level, not a rigorous accuracy check.

Usage (with farm_sim.launch.py already running):
  ros2 run farmsim_perception live_segmentation_view
or automatically via:
  ros2 launch farmsim_farm farm_sim.launch.py live_view:=true follow:=true rows_file:=...
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from farmsim_farm.segment_rows import vegetation_mask

WINDOW = 'Thorvald - camera + segmentation'


class LiveSegmentationView(Node):

    def __init__(self):
        super().__init__('live_segmentation_view')
        self.declare_parameter('hue_min', 36)
        self.declare_parameter('sat_min', 40)
        self._hue_min = self.get_parameter('hue_min').get_parameter_value().integer_value
        self._sat_min = self.get_parameter('sat_min').get_parameter_value().integer_value

        self._bridge = CvBridge()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self.create_subscription(Image, '/front_camera', self._image_cb, 10)
        self.get_logger().info('Waiting for /front_camera frames...')

    def _image_cb(self, msg):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        mask = vegetation_mask(frame, hue_min=self._hue_min, sat_min=self._sat_min)

        overlay = frame.copy()
        green_tint = np.zeros_like(overlay)
        green_tint[:, :, 1] = 255
        veg = mask > 0
        overlay[veg] = cv2.addWeighted(overlay, 0.6, green_tint, 0.4, 0)[veg]

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([overlay, mask_bgr])
        cv2.imshow(WINDOW, combined)
        # Pumps the HighGUI event loop / redraws the window. Called here,
        # inside the subscription callback under the default single-
        # threaded spin, rather than off a separate timer/thread -- HighGUI
        # isn't thread-safe, and one imshow+waitKey per camera frame is
        # simplest.
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LiveSegmentationView()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
