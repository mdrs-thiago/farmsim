"""
video.py

Camera videos of control runs (run_control.py run.video=true).

  record   subscribes to /front_camera and /chase_camera and writes
           <out_dir>/{front,chase}_camera.mp4 plus the sim-time stamp of every
           frame (*_stamps.json) until SIGINT/SIGTERM. Started by run_control
           next to the sim; the chase camera comes from gz_camera_chase.xacro.
  compose  one video per run (video.mp4): chase camera on top, front camera
           below, with s, v, omega, offset and heading from control_log.jsonl
           drawn on it. With several runs the tiles sit side by side, synced
           on time since each run's first controller step.

  python -m farmsim_utils.video record OUT_DIR
  python -m farmsim_utils.video compose OUT.mp4 RUN_DIR [RUN_DIR ...] [--speedup 4]
"""

import argparse
import json
import signal
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ('chase_camera', 'front_camera')
TILE_W, CHASE_H, FRONT_H = 960, 540, 540
COLORS = {'pd': (219, 111, 42), 'fuzzy': (12, 89, 232)}  # BGR


def record(out_dir):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    out_dir = Path(out_dir)
    writers, stamps = {}, {name: [] for name in CAMERAS}

    def on_image(name, msg):
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., :3]
        if name not in writers:
            # Stamps carry the timing; the nominal fps only matters for plain playback.
            writers[name] = cv2.VideoWriter(str(out_dir / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                            20.0, (msg.width, msg.height))
        writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else frame)
        stamps[name].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    rclpy.init()
    node = rclpy.create_node('run_video_recorder')
    for name in CAMERAS:
        node.create_subscription(Image, '/' + name, lambda m, n=name: on_image(n, m), qos_profile_sensor_data)
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.2)
    for name, w in writers.items():
        w.release()
        with open(out_dir / f'{name}_stamps.json', 'w') as f:
            json.dump(stamps[name], f)
        print(f'{name}: {len(stamps[name])} frames')
    node.destroy_node()
    rclpy.shutdown()


class _Stream:
    """Sequential reader returning the latest frame stamped <= t (black before the first)."""

    def __init__(self, path, size):
        self.size = size
        self.frame = np.zeros((size[1], size[0], 3), np.uint8)
        stamps_file = path.with_name(path.stem + '_stamps.json')
        self.cap = cv2.VideoCapture(str(path)) if path.exists() and stamps_file.exists() else None
        self.stamps = json.loads(stamps_file.read_text()) if self.cap else []
        self.i = -1

    def at(self, t):
        while self.i + 1 < len(self.stamps) and self.stamps[self.i + 1] <= t:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frame = cv2.resize(frame, self.size)
            self.i += 1
        return self.frame


def _text(img, lines, color, origin=(12, 30)):
    x, y = origin
    cv2.rectangle(img, (0, 0), (480, y + 28 * len(lines) - 14), (0, 0, 0), -1)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (x, y + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    color if k == 0 else (255, 255, 255), 2 if k == 0 else 1, cv2.LINE_AA)


def compose(out_mp4, run_dirs, speedup=4.0, fps=25):
    runs = []
    for d in map(Path, run_dirs):
        log = [json.loads(line) for line in open(d / 'control_log.jsonl')]
        cfg = (d / 'control.yaml').read_text() if (d / 'control.yaml').exists() else ''
        strategy = next((line.split(':')[1].split('#')[0].strip() for line in cfg.splitlines()
                         if line.strip().startswith('strategy:')), d.name)
        lt = np.array([r['t'] for r in log])
        runs.append(dict(name=d.name, strategy=strategy, log=log, start=lt[0], lt=lt - lt[0],
                         chase=_Stream(d / 'chase_camera.mp4', (TILE_W, CHASE_H)),
                         front=_Stream(d / 'front_camera.mp4', (FRONT_H * 4 // 3, FRONT_H))))
    size = (TILE_W * len(runs), CHASE_H + FRONT_H)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    t_len = max(r['lt'][-1] for r in runs) + 2.0
    for tau in np.arange(0.0, t_len, speedup / fps):
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        for col, r in enumerate(runs):
            x0 = col * TILE_W
            tile = canvas[:, x0:x0 + TILE_W]
            tile[:CHASE_H] = r['chase'].at(r['start'] + tau)
            front = r['front'].at(r['start'] + tau)
            pad = (TILE_W - front.shape[1]) // 2
            tile[CHASE_H:, pad:pad + front.shape[1]] = front
            i = int(np.clip(np.searchsorted(r['lt'], tau), 0, len(r['log']) - 1))
            row = r['log'][i]
            status = 'finished' if tau > r['lt'][-1] else ('in row' if row['kind'] == 'row' else 'lead-in / out')
            _text(tile, [f"{r['strategy'].upper()}  t = {tau:5.1f} s   ({r['name']})",
                         f"s = {row['s']:5.1f} / {row['total']:.1f} m  [{status}]",
                         f"v = {row['v']:.2f} m/s   w = {row['w']:+.3f} rad/s",
                         f"offset = {row['e_gt'] * 100:+5.1f} cm  heading = {np.degrees(row['psi_gt']):+5.1f} deg"],
                  COLORS.get(r['strategy'], (255, 255, 255)))
            cv2.putText(tile, 'front camera', (12, CHASE_H + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if col:
                cv2.line(canvas, (x0, 0), (x0, size[1]), (255, 255, 255), 3)
        cv2.putText(canvas, f'{speedup:g}x', (size[0] - 60, size[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()
    return out_mp4


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('out_dir')
    c = sub.add_parser('compose')
    c.add_argument('out_mp4')
    c.add_argument('run_dirs', nargs='+')
    c.add_argument('--speedup', type=float, default=4.0)
    a = p.parse_args()
    if a.cmd == 'record':
        record(a.out_dir)
    else:
        print(compose(a.out_mp4, a.run_dirs, a.speedup))


if __name__ == '__main__':
    main()
