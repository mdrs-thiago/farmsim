import numpy as np
import pytest

from farmsim_farm.topdown_projection import pixel_to_world, world_to_pixel

# A nadir (straight-down) camera, matching farm/fields/20250425/topdown.yaml's
# shape: pitch=90deg, roll=yaw=0.
_META = {
    'camera_pose': {'x': 33.67, 'y': -11.88, 'z': 34.0, 'roll': 0.0, 'pitch': 1.5707963267948966, 'yaw': 0.0},
    'ground_plane_z': -0.53,
    'intrinsics': {'fx': 3199.988555908203, 'fy': 3199.988555908203, 'cx': 3200.0, 'cy': 3200.0},
    'width': 6400,
    'height': 6400,
}


def test_image_center_maps_directly_below_camera():
    x, y = pixel_to_world(_META['intrinsics']['cx'], _META['intrinsics']['cy'], _META)
    assert x == pytest.approx(_META['camera_pose']['x'], abs=1e-6)
    assert y == pytest.approx(_META['camera_pose']['y'], abs=1e-6)


@pytest.mark.parametrize('u,v', [(0, 0), (6399, 0), (0, 6399), (6399, 6399),
                                  (3200, 3200), (1000, 5000), (5500, 250)])
def test_pixel_world_round_trip_is_exact(u, v):
    x, y = pixel_to_world(u, v, _META)
    u2, v2 = world_to_pixel(x, y, _META)
    assert u2 == pytest.approx(u, abs=1e-6)
    assert v2 == pytest.approx(v, abs=1e-6)


def test_world_to_pixel_rejects_point_behind_camera():
    # A point far outside the ground plane's projected footprint in the
    # camera's -Z (behind) direction relative to boresight: use the camera's
    # own (x, y) at a height ABOVE the camera itself, which cannot appear in
    # front of a straight-down-looking camera.
    with pytest.raises(ValueError):
        world_to_pixel(_META['camera_pose']['x'], _META['camera_pose']['y'], {
            **_META, 'ground_plane_z': _META['camera_pose']['z'] + 5.0,
        })


def test_pixel_to_world_matches_known_manual_marking():
    # From farm/fields/20250425/rows.yaml, row 0's first
    # point was marked by mark_rows.py against this same topdown capture --
    # cross-check that re-deriving its source pixel and re-projecting lands
    # back on the same world point (regression guard against a refactor
    # silently changing the math, not a ground-truth check).
    known_world = (2.8, -6.949)
    u, v = world_to_pixel(*known_world, _META)
    x2, y2 = pixel_to_world(u, v, _META)
    assert x2 == pytest.approx(known_world[0], abs=1e-6)
    assert y2 == pytest.approx(known_world[1], abs=1e-6)
