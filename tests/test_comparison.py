"""Check physical error definitions and reference playback compatibility."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/upstream-git/breezewrf-supermimic'


class TrackingMetricTests(unittest.TestCase):
    def test_world_drift_does_not_disappear_into_local_pose_metrics(self):
        from g1_supermimic.runtime import activate
        activate()
        from g1_supermimic.diagnostics import tracking_errors
        from mjlab.utils.lab_api.math import quat_apply
        positions = torch.tensor([[[0., 0., .8], [.2, .1, 1.]]])
        identity = torch.tensor([[[1., 0., 0., 0.]]]).expand(1, 2, 4)
        rotated = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)]]]).expand(1, 2, 4)
        robot = positions[:, :1] + torch.tensor([2., 0., 0.]) + quat_apply(rotated, positions - positions[:, :1])
        cmd = SimpleNamespace(body_pos_w=positions, body_quat_w=identity,
            anchor_pos_w=positions[:, 0], anchor_quat_w=identity[:, 0],
            robot_body_pos_w=robot, robot_body_quat_w=rotated,
            robot_anchor_pos_w=robot[:, 0], robot_anchor_quat_w=rotated[:, 0],
            joint_pos=torch.zeros(1, 29), robot_joint_pos=torch.zeros(1, 29))
        result = tracking_errors(cmd)
        self.assertAlmostEqual(result['anchor_xy_m'].item(), 2.)
        self.assertAlmostEqual(result['anchor_yaw_deg'].item(), 90., places=4)
        self.assertLess(result['body_aligned_m'].item(), 1e-6)
        self.assertLess(result['body_aligned_rotation_deg'].item(), 1e-4)
        self.assertGreater(result['body_world_m'].item(), 1.)


@unittest.skipUnless(SOURCE.is_dir(), 'Pinned Breezewrf checkout not present')
class BreezewrfPlaybackTests(unittest.TestCase):
    def test_compatibility_preserves_foot_contact_priorities(self):
        sys.path.insert(0, str(SOURCE))
        from g1_supermimic.reference_breezewrf import configure_compatibility
        configure_compatibility()
        from src.tasks.g1_supermimic.core.config.robot_cfg import get_supermimic_robot_cfg
        from mjlab.entity import Entity
        robot = Entity(get_supermimic_robot_cfg())
        geoms = [g for g in robot.spec.geoms if g.name.endswith('_collision')]
        feet = [g for g in geoms if '_foot' in g.name]
        self.assertEqual(len(feet), 8)
        self.assertTrue(all(g.priority == 1 and g.condim == 3 for g in feet))
        self.assertTrue(all(g.priority == 0 for g in geoms if g not in feet))


if __name__ == '__main__':
    unittest.main()
