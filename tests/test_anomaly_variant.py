"""Geometry and mask invariants for world-anchored visual anomalies."""

import unittest

import numpy as np

from farmsim_perception.anomalies.make_anomaly_variant import apply_anomalies, make_anomalies, project_polygon


class AnomalyVariantTest(unittest.TestCase):
    def test_projection_and_mask_match_edited_ground_pixels(self):
        # Camera at z=2 looking straight down: world x goes right in the image.
        transform = np.diag([1.0, -1.0, -1.0, 1.0])
        transform[:3, 3] = (0.0, 0.0, 2.0)
        camera = {'fx': 32.0, 'fy': 32.0, 'cx': 32.0, 'cy': 32.0,
                  'width': 64, 'height': 64}
        polygon = [[-0.8, -0.8, 0.0], [0.8, -0.8, 0.0],
                   [0.8, 0.8, 0.0], [-0.8, 0.8, 0.0]]
        projected = project_polygon(polygon, transform, camera, 64, 64)
        self.assertTrue(np.all(projected >= 0))
        self.assertTrue(np.all(projected < 64))

        images = np.full((1, 64, 64, 3), 220, np.uint8)
        semantic = np.full((1, 64, 64), 2, np.uint8)
        semantic[:, :, :32] = 1  # crop canopy: never paint over it
        anomalies = [{'kind': 1, 'vertices_world': polygon}]
        edited, mask = apply_anomalies(images, semantic,
                                       [{'T_world_optical': transform}], camera, anomalies)
        self.assertGreater(int((mask != 0).sum()), 0)
        self.assertTrue(np.all(mask[:, :, :32] == 0))
        self.assertTrue(np.array_equal(edited[mask == 0], images[mask == 0]))
        self.assertTrue(np.all(np.any(edited[mask != 0] != images[mask != 0], axis=1)))

    def test_map_generation_is_seeded_and_on_terrain(self):
        # Forward optical +Z is world +X; camera travels on a flat surface.
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        transform[:3, 3] = (1.0, 1.0, 1.0)
        poses = [{'T_world_optical': transform} for _ in range(20)]
        heightmap = {'heights': np.zeros((100, 100), np.float32),
                     'x0': 0.0, 'y0': 0.0, 'resolution': 0.1}
        a = make_anomalies(poses, heightmap, seed=42, count=2)
        b = make_anomalies(poses, heightmap, seed=42, count=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 2)
        for anomaly in a:
            self.assertTrue(np.allclose(np.asarray(anomaly['vertices_world'])[:, 2], 0.015))


if __name__ == '__main__':
    unittest.main()
