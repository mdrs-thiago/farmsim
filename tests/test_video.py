import json

import cv2
import numpy as np

from farmsim_utils.video import compose


def _fake_camera(path, size, stamps):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, size)
    for k in range(len(stamps)):
        w.write(np.full((size[1], size[0], 3), 20 * k % 255, np.uint8))
    w.release()
    path.with_name(path.stem + '_stamps.json').write_text(json.dumps(stamps))


def _fake_run(d, strategy):
    d.mkdir()
    (d / 'control.yaml').write_text(f'control:\n  strategy: {strategy}\n')
    with open(d / 'control_log.jsonl', 'w') as f:
        for k in range(40):
            f.write(json.dumps({'t': 10.0 + 0.05 * k, 'kind': 'row', 's': 0.01 * k, 'total': 5.0,
                                'v': 0.3, 'w': 0.0, 'e_gt': 0.01, 'psi_gt': 0.0}) + '\n')
    _fake_camera(d / 'chase_camera.mp4', (320, 180), [10.0 + 0.1 * k for k in range(20)])


def test_compose_side_by_side(tmp_path):
    _fake_run(tmp_path / 'a', 'pd')
    _fake_run(tmp_path / 'b', 'fuzzy')  # no front camera: drawn black
    out = compose(tmp_path / 'v.mp4', [tmp_path / 'a', tmp_path / 'b'], speedup=1.0, fps=10)
    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 2 * 960 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1080
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 20
