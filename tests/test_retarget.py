"""Check retarget import conventions with analytic translation/rotation trajectories."""
import tempfile
import unittest
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.core.data.motion_convert import G1_JOINT_NAMES, load_g1_model
from g1_supermimic.retarget import import_motion, source_joint_names, validate_csv


class RetargetImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_g1_model()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        # Reverse the source qpos order to exercise the name-based mapping.
        self.names = list(reversed(G1_JOINT_NAMES))
        self.xml = self.root / 'g1.xml'
        self.xml.write_text('<mujoco><worldbody><body name="pelvis"><freejoint/>'
                            + ''.join(f'<body><joint name="{name}"/></body>' for name in self.names)
                            + '</body></worldbody></mujoco>')
        times = np.arange(31) / 30
        self.raw = np.zeros((31, 36))
        self.raw[:, 1] = .6 * times
        self.raw[:, 2] = .78
        angle = .3 + .4 * times
        self.raw[:, 5] = np.sin(angle / 2)
        self.raw[:, 6] = np.cos(angle / 2)
        self.raw[1::2, 3:7] *= -1  # The orientation has no jumps.
        ranges = self.model.jnt_range[1:]
        neutral = np.clip(np.zeros(29), ranges[:, 0], ranges[:, 1])
        self.raw[:, 7:] = neutral[[G1_JOINT_NAMES.index(n) for n in self.names]]
        self.knee = 7 + self.names.index('left_knee_joint')
        self.raw[:, self.knee] = .1 + .2 * times
        self.csv = self.root / 'motion.csv'
        self.output = self.root / 'motion.npz'
        self.save_csv()

    def save_csv(self):
        np.savetxt(self.csv, self.raw, delimiter=',')

    def test_slowing_preserves_distance_and_halves_velocities(self):
        report = import_motion(self.csv, self.xml, self.output, 30., .5)
        self.assertEqual(report['frames'], 101)
        self.assertEqual(report['duration_seconds'], 2.)
        np.testing.assert_allclose(report['root_displacement_m'], [0., .6, 0.], atol=1e-7)
        with np.load(self.output, allow_pickle=False) as motion:
            self.assertEqual(motion['joint_names'].tolist(), list(G1_JOINT_NAMES))
            pelvis = motion['body_names'].tolist().index('pelvis')
            np.testing.assert_allclose(motion['body_lin_vel_w'][:, pelvis, 1], .3, atol=1e-6)
            np.testing.assert_allclose(motion['body_ang_vel_w'][:, pelvis, 2], .2, atol=1e-6)
            np.testing.assert_allclose(motion['body_quat_w'][0, pelvis],
                                       [np.cos(.15), 0, 0, np.sin(.15)], atol=1e-7)
            knee = list(G1_JOINT_NAMES).index('left_knee_joint')
            np.testing.assert_allclose(motion['joint_vel'][:, knee], .1, atol=1e-6)
            np.testing.assert_allclose(motion['joint_pos'][[0, -1], knee], [.1, .3], atol=1e-7)
        self.assertTrue(self.output.with_suffix('.json').is_file())
        self.assertEqual(report['quaternion'], 'wxyz')

    def test_invalid_layout_rate_and_quaternions_are_rejected(self):
        for fps, speed in [(0, 1), (30, 0), (np.nan, 1), (30, np.inf), (1e-300, 1e-300)]:
            with self.subTest(fps=fps, speed=speed), self.assertRaises(ValueError):
                validate_csv(self.raw, fps, speed)
        for raw in (self.raw[:, :35], self.raw[:1], np.zeros((5, 36)), np.full((5, 36), np.nan)):
            with self.assertRaises(ValueError):
                validate_csv(raw, 30., 1.)

    def test_input_limit_violation_cannot_be_hidden_by_downsampling(self):
        self.raw[1, self.knee] = 20.
        self.save_csv()
        with self.assertRaisesRegex(ValueError, 'Joint limit violation: frame 1'):
            import_motion(self.csv, self.xml, self.output, 200.)
        self.assertFalse(self.output.exists())

    def test_existing_motion_or_metadata_is_preserved(self):
        for path in (self.output, self.output.with_suffix('.json')):
            path.write_text('keep')
            with self.assertRaises(FileExistsError):
                import_motion(self.csv, self.xml, self.output, 30.)
            self.assertEqual(path.read_text(), 'keep')
            path.unlink()

    def test_declared_sdk_order_works_without_source_xml(self):
        self.raw[:, 7:] = self.raw[:, 7:][:, [self.names.index(n) for n in G1_JOINT_NAMES]]
        self.save_csv()
        report = import_motion(self.csv, None, self.output, 30.)
        self.assertEqual(report['input_joint_order'], 'declared_g1_sdk')
        self.assertIsNone(report['source_xml'])
        self.assertEqual(report['joint_permutation'], list(range(29)))

    def test_unseen_motion_inference_is_explicit_and_does_not_change_resume_contract(self):
        from g1_supermimic.inference import check_inference_data
        from g1_supermimic.checkpoints import check_data_contract
        (self.root / 'run_manifest.json').write_text(json.dumps({'data': [{'sha256': 'old motion'}]}))
        checkpoint = self.root / 'best.pt'
        with self.assertRaisesRegex(ValueError, 'hashes/order'):
            check_inference_data(SimpleNamespace(), checkpoint, [self.csv])
        with self.assertRaisesRegex(ValueError, 'explicit --motion or --npz-dir'):
            check_inference_data(SimpleNamespace(allow_new_motion=True), checkpoint, [self.csv])
        check_inference_data(SimpleNamespace(allow_new_motion=True, npz_dir=self.root), checkpoint, [self.csv])
        check_inference_data(SimpleNamespace(allow_new_motion=True, motion=self.csv), checkpoint, [self.csv])
        with self.assertRaisesRegex(ValueError, 'hashes/order'):
            check_data_contract(checkpoint, [self.csv])

    def test_xml_order_is_explicit_and_unknown_models_are_rejected(self):
        self.assertEqual(source_joint_names(self.xml), self.names)
        self.xml.write_text(self.xml.read_text().replace(self.names[0], 'unknown_joint'))
        with self.assertRaisesRegex(ValueError, 'joint names differ'):
            import_motion(self.csv, self.xml, self.output, 30.)
        self.xml.write_text('<mujoco><include file="other.xml"/></mujoco>')
        with self.assertRaisesRegex(ValueError, 'no includes'):
            source_joint_names(self.xml)


if __name__ == '__main__':
    unittest.main()
