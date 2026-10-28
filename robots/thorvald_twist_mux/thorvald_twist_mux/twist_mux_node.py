"""
Priority-based Twist multiplexer.

Reads topics and locks from a YAML parameter list. Each topic entry has:
  name, topic, timeout, priority
A lock entry has:
  name, topic, timeout, priority  (Bool — when True, blocks all equal/lower priorities)

The highest-priority non-timed-out, non-locked source wins.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class TwistSource:
    def __init__(self, name, topic, timeout, priority):
        self.name = name
        self.topic = topic
        self.timeout = timeout       # seconds; 0 = never timeout
        self.priority = priority
        self.last_msg = None
        self.last_stamp = None       # rclpy.time.Time

    def is_active(self, now: Time) -> bool:
        if self.last_msg is None:
            return False
        if self.timeout <= 0.0:
            return True
        age = (now - self.last_stamp).nanoseconds * 1e-9
        return age < self.timeout


class LockSource:
    def __init__(self, name, topic, timeout, priority):
        self.name = name
        self.topic = topic
        self.timeout = timeout
        self.priority = priority
        self.locked = False
        self.last_stamp = None

    def is_active(self, now: Time) -> bool:
        if self.last_stamp is None:
            return False
        if self.timeout <= 0.0:
            return self.locked
        age = (now - self.last_stamp).nanoseconds * 1e-9
        return self.locked and age < self.timeout


class TwistMuxNode(Node):

    def __init__(self):
        super().__init__('twist_mux')

        # Load topics list
        self.declare_parameter('topics', rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter('locks', rclpy.Parameter.Type.STRING_ARRAY)

        topic_names = self._get_param_list('topics')
        lock_names  = self._get_param_list('locks')

        self._sources: list[TwistSource] = []
        self._locks:   list[LockSource]  = []
        self._twist_subs = []
        self._lock_subs  = []

        for name in topic_names:
            pfx = f'topic_{name}'
            self.declare_parameter(f'{pfx}.topic',    f'/{name}/cmd_vel')
            self.declare_parameter(f'{pfx}.timeout',  0.5)
            self.declare_parameter(f'{pfx}.priority', 10)
            topic    = self.get_parameter(f'{pfx}.topic').get_parameter_value().string_value
            timeout  = self.get_parameter(f'{pfx}.timeout').get_parameter_value().double_value
            priority = self.get_parameter(f'{pfx}.priority').get_parameter_value().integer_value
            src = TwistSource(name, topic, timeout, priority)
            self._sources.append(src)
            sub = self.create_subscription(
                Twist, topic,
                lambda msg, s=src: self._twist_cb(msg, s), 10)
            self._twist_subs.append(sub)
            self.get_logger().info(f'Twist source: {name!r} on {topic!r} '
                                   f'pri={priority} timeout={timeout}s')

        for name in lock_names:
            pfx = f'lock_{name}'
            self.declare_parameter(f'{pfx}.topic',    f'/{name}/joy_priority')
            self.declare_parameter(f'{pfx}.timeout',  0.5)
            self.declare_parameter(f'{pfx}.priority', 10)
            topic    = self.get_parameter(f'{pfx}.topic').get_parameter_value().string_value
            timeout  = self.get_parameter(f'{pfx}.timeout').get_parameter_value().double_value
            priority = self.get_parameter(f'{pfx}.priority').get_parameter_value().integer_value
            lk = LockSource(name, topic, timeout, priority)
            self._locks.append(lk)
            sub = self.create_subscription(
                Bool, topic,
                lambda msg, l=lk: self._lock_cb(msg, l), 10)
            self._lock_subs.append(sub)
            self.get_logger().info(f'Lock source: {name!r} on {topic!r} '
                                   f'pri={priority} timeout={timeout}s')

        self._pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.02, self._publish_best)   # 50 Hz

    # ------------------------------------------------------------------ helpers

    def _get_param_list(self, name: str):
        try:
            return self.get_parameter(name).get_parameter_value().string_array_value
        except Exception:
            return []

    def _twist_cb(self, msg: Twist, src: TwistSource):
        src.last_msg   = msg
        src.last_stamp = self.get_clock().now()

    def _lock_cb(self, msg: Bool, lk: LockSource):
        lk.locked     = msg.data
        lk.last_stamp = self.get_clock().now()

    def _blocked_at_or_below(self, priority: int, now: Time) -> bool:
        for lk in self._locks:
            if lk.priority >= priority and lk.is_active(now):
                return True
        return False

    def _publish_best(self):
        now = self.get_clock().now()
        best: TwistSource | None = None

        for src in sorted(self._sources, key=lambda s: s.priority, reverse=True):
            if not src.is_active(now):
                continue
            if self._blocked_at_or_below(src.priority, now):
                continue
            best = src
            break

        if best is not None:
            self._pub.publish(best.last_msg)
        else:
            self._pub.publish(Twist())   # zero velocity


def main(args=None):
    rclpy.init(args=args)
    node = TwistMuxNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
