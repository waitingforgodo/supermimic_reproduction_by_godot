"""源码替换的隔离契约：不把旧 checkpoint 或换过的动作交给新网络。"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_supermimic import checkpoints as bridge
from g1_supermimic.configuration import merge_config
from g1_supermimic.inference import RewardManagerView


class ReproducedTests(unittest.TestCase):
    def motion_fixture(self, root):
        import numpy as np
        from g1_supermimic.support.contract import SDK_JOINTS
        from g1_supermimic.runtime import activate
        activate()
        from g1_supermimic.core.envs.tracking_env_cfg import TRACKED_BODIES
        path = root/'clip.npz'
        np.savez(path, fps=50., joint_names=np.array(SDK_JOINTS),
                 body_names=np.array(TRACKED_BODIES), joint_pos=np.zeros((3, 29)),
                 joint_vel=np.zeros((3, 29)), body_pos_w=np.zeros((3, 14, 3)),
                 body_quat_w=np.tile([1., 0, 0, 0], (3, 14, 1)),
                 body_lin_vel_w=np.zeros((3, 14, 3)), body_ang_vel_w=np.zeros((3, 14, 3)))
        return path

    def test_checkpoint_alias_restores_workload_without_applying_fresh_profile(self):
        import torch
        from dataclasses import asdict
        from g1_supermimic.cli import parser
        from g1_supermimic.training import configure
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.motion_fixture(root)
            from g1_supermimic.variants import TrainConfig
            cfg = TrainConfig()
            cfg.data.npz_dir, cfg.data.clips = root, ['clip']
            cfg.env.num_envs = 17
            cfg.policy_trainer.horizon, cfg.policy_trainer.batch_size = 32, 256
            cfg.wm_trainer.batch_size = 64
            checkpoint = root/'checkpoint.pt'
            torch.save(dict(config=asdict(cfg), iteration=10, wm={}, policy={}, critic={}), checkpoint)
            args = parser().parse_args(['resume', '--checkpoint', str(checkpoint), '--iterations', '100'])
            self.assertIsNone(args.resume)
            restored, manifest = configure(args)
            self.assertEqual(restored.resume, checkpoint)
            self.assertEqual(restored.env.num_envs, 17)
            self.assertEqual(restored.policy_trainer.horizon, 32)
            self.assertEqual(restored.policy_trainer.batch_size, 256)
            self.assertEqual(restored.wm_trainer.batch_size, 64)
            self.assertFalse(restored.optimizations.actor_input_normalization)
            self.assertEqual(manifest['configuration_base'], 'checkpoint')

    def test_fresh_run_applies_local_profile_before_partial_config_and_cli(self):
        from g1_supermimic.cli import parser
        from g1_supermimic.training import configure
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            motion = self.motion_fixture(root)
            override = root/'override.json'
            override.write_text(json.dumps({'policy_trainer': {'horizon': 20, 'batch_size': 128}}))
            args = parser().parse_args(['train', '--motion', str(motion), '--config', str(override),
                                       '--policy-batch-size', '64'])
            cfg, manifest = configure(args)
            self.assertEqual(cfg.env.num_envs, 4096)
            self.assertEqual(cfg.wm_trainer.batch_size, 512)
            self.assertIsNone(cfg.wm_trainer.compile)
            self.assertEqual(cfg.policy_trainer.horizon, 20)
            self.assertEqual(cfg.policy_trainer.batch_size, 64)
            self.assertTrue(cfg.optimizations.actor_input_normalization)
            self.assertTrue(cfg.model_alignment.cached_body_reference)
            self.assertEqual(manifest['configuration_base'], 'lafan1_4096_h16_eager')

    def test_reward_rate_conversion_uses_control_dt_only_when_enabled(self):
        from types import SimpleNamespace
        manager = SimpleNamespace(_step_reward=3., _scale_by_dt=True, active_terms=['pose'])
        proxy = RewardManagerView(manager, .02)
        self.assertAlmostEqual(proxy.term_rewards, .06)
        self.assertEqual(proxy.active_terms, ['pose'])
        manager._scale_by_dt = False
        self.assertEqual(proxy.term_rewards, 3.)

    def test_latest_never_searches_legacy_run_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder) / 'new'
            old = Path(folder) / 'old'
            old.mkdir()
            (old / 'best.pt').touch()
            with patch.object(bridge, 'RUNS', runs):
                with self.assertRaisesRegex(FileNotFoundError, 'No Godot-SuperMimic'):
                    bridge.resolve_checkpoint('latest')
                (runs / 'run').mkdir(parents=True)
                ckpt = runs / 'run/ckpt_latest.pt'
                ckpt.touch()
                self.assertEqual(bridge.resolve_checkpoint('latest'), ckpt)

    def test_unsupported_manifest_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            path.touch()
            (path.parent / 'run_manifest.json').write_text(json.dumps(dict(implementation='unsupported_method')))
            with self.assertRaisesRegex(ValueError, 'Unsupported checkpoint implementation'):
                bridge.resolve_checkpoint(path)

    def test_reference_contract_rejects_reordering_and_modified_data(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            a, b = folder / 'a.npz', folder / 'b.npz'
            a.write_bytes(b'first clip')
            b.write_bytes(b'second clip')
            manifest = dict(data=[dict(sha256=bridge.digest(p)) for p in (a, b)])
            (folder / 'run_manifest.json').write_text(json.dumps(manifest))
            bridge.check_data_contract(folder / 'ckpt_latest.pt', [a, b])
            with self.assertRaisesRegex(ValueError, 'hashes/order'):
                bridge.check_data_contract(folder / 'ckpt_latest.pt', [b, a])
            a.write_bytes(b'changed choreography')
            with self.assertRaisesRegex(ValueError, 'hashes/order'):
                bridge.check_data_contract(folder / 'ckpt_latest.pt', [a, b])

    def test_nested_config_override_rejects_typos(self):
        @dataclass
        class Schedule:
            total_iterations: int = 320000
        @dataclass
        class Config:
            schedule: Schedule = field(default_factory=Schedule)
        cfg = merge_config(Config(), dict(schedule=dict(total_iterations=3000)))
        self.assertEqual(cfg.schedule.total_iterations, 3000)
        with self.assertRaisesRegex(ValueError, 'Unknown Schedule field'):
            merge_config(cfg, dict(schedule=dict(total_iteraions=3000)))


if __name__ == '__main__':
    unittest.main()
