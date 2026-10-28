"""Pure camera-mount geometry shared by recording and offline label tools.

This module deliberately has no ROS imports.  Keeping these small rigid-body
operations importable offline lets metadata repairs use exactly the same math
as the recorder/exporter instead of maintaining a second implementation.
"""

import numpy as np


# Sensor (REP-103: X fwd/Y left/Z up) -> optical (Z fwd/X right/Y down).
R_SENSOR_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def rpy_matrix(roll, pitch, yaw):
    """URDF/SDF fixed-axis rpy -> rotation matrix: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
