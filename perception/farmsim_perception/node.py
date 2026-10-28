"""
node.py  (ros2 run farmsim_perception perception_node)

Runs the row-perception CNN on every front-camera frame and publishes the
row state for the controller on /perception/row_state
(farmsim_utils.row_state documents the field order).

Parameters:
  checkpoint   path to a row_net checkpoint (last.pt)
  device       'auto' | 'cpu' | 'cuda'
  image_topic  default /front_camera
  show         open an OpenCV window with the segmentation and row line
"""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from farmsim_perception.inference import RowPerception
from farmsim_utils import row_state


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('image_topic', '/front_camera')
        self.declare_parameter('show', False)
        ckpt = self.get_parameter('checkpoint').value
        if not ckpt:
            raise RuntimeError('checkpoint parameter is required')
        self.perception = RowPerception(ckpt, self.get_parameter('device').value)
        self.show = self.get_parameter('show').value
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Float32MultiArray, row_state.TOPIC, 10)
        # depth 1: always process the newest frame, never a backlog
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image, 1)
        self.get_logger().info(f'model={self.perception.model.kind} device={self.perception.device}')

    def on_image(self, msg):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        est = self.perception(rgb)
        self.pub.publish(row_state.encode(Float32MultiArray, **est))
        if self.show:
            self._draw(rgb, est)

    def _draw(self, rgb, est):
        import cv2
        h, w = rgb.shape[:2]
        colors = np.array([(0, 0, 0), (40, 200, 40), (40, 90, 255), (255, 200, 0)], np.uint8)
        seg = cv2.resize(colors[est['seg']], (w, h), interpolation=cv2.INTER_NEAREST)
        img = cv2.addWeighted(rgb, 0.6, seg, 0.4, 0)
        pts = [(int((est[f'u_{k}'] + 1) * w / 2), int((est[f'v_{k}'] + 1) * h / 2)) for k in ('near', 'far')]
        cv2.line(img, pts[0], pts[1], (255, 0, 0), 2)
        cv2.putText(img, f"e={est['lateral_offset']:+.2f}m psi={np.degrees(est['heading_error']):+.1f}deg "
                         f"conf={est['confidence']:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('perception', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


def main(args=None):
    try:
        with rclpy.init(args=args):
            rclpy.spin(PerceptionNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
