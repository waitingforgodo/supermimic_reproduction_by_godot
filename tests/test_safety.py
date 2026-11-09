"""Strict upstream settings and numerical safeguards that preserve valid updates."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from g1_supermimic.checkpoints import load_checkpoint
from g1_supermimic.configuration import train_parser
from g1_supermimic.safety import guard_training, trainer_state
from g1_supermimic.strict import LAFAN_DATASET, training_contract


class BackendCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(torch.__version__.startswith('2.9.'), 'PyTorch 2.9 regression')
    def test_mjlab_configuration_remains_readable_by_inductor(self):
        # Isolate global backend state from the other numerical tests. The legacy
        # getter below is exactly what raised before the first compiled update.
        result = subprocess.run([sys.executable, '-c', '''
import torch
from g1_supermimic.runtime import configure_backend_compatibility
assert configure_backend_compatibility()
from mjlab.utils.torch import configure_torch_backends
configure_torch_backends()
assert torch.backends.cuda.matmul.allow_tf32 is True
assert torch.backends.cuda.matmul.fp32_precision == 'tf32'
assert torch.backends.cudnn.allow_tf32 is True
assert torch.backends.cudnn.benchmark is True
configure_torch_backends(allow_tf32=False, deterministic=True)
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cuda.matmul.fp32_precision == 'ieee'
assert torch.backends.cudnn.allow_tf32 is False
assert torch.backends.cudnn.benchmark is False
assert torch.backends.cudnn.deterministic is True
'''], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@dataclass
class Data:
    npz_dir: str = 'upstream/data'
    clips: list | None = None


@dataclass
class Settings:
    data: Data = field(default_factory=Data)
    batch_size: int = 4096
    run_name: str = 'debug'
    log_root: str = 'logs'
    device: str = 'cuda:0'
    resume: str | None = None


class StrictSettingsTests(unittest.TestCase):
    def test_local_default_and_explicit_strict_mode_are_unambiguous(self):
        self.assertFalse(train_parser().parse_args([]).strict_upstream)
        self.assertTrue(train_parser().parse_args(['--strict-upstream']).strict_upstream)
        self.assertFalse(train_parser().parse_args(['--allow-config-changes']).strict_upstream)

    def test_strict_refuses_batch_changes_and_records_custom_changes(self):
        cfg = Settings(batch_size=128)
        with self.assertRaisesRegex(ValueError, 'batch_size.*128.*4096'):
            training_contract(cfg, Settings(), [], strict=True)
        report = training_contract(cfg, Settings(), [], strict=False)
        self.assertFalse(report['upstream_parameters_match'])
        self.assertEqual(report['parameter_differences']['batch_size'], dict(upstream=4096, effective=128))

    def test_strict_checks_all_forty_clip_hashes_but_allows_path_remapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = [root / f'clip_{i}.npz' for i in range(40)]
            for path in files:
                path.write_bytes(b'pinned clip')
            metadata = dict(dataset=LAFAN_DATASET, clips=40, revision='pinned',
                            outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            (root / 'dataset_manifest.json').write_text(json.dumps(metadata))
            cfg = Settings(data=Data(str(root)), run_name='my_run', log_root='outputs')
            self.assertTrue(training_contract(cfg, Settings(), files)['upstream_parameters_match'])
            files[0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'data differs'):
                training_contract(cfg, Settings(), files)


def tiny_trainer(folder):
    class Update:
        def __init__(self, module):
            self.module = module
            self.opt = torch.optim.SGD(module.parameters(), lr=.01)

        def update(self, buffer=None):
            self.opt.zero_grad(set_to_none=True)
            loss = self.module(torch.ones(3, 2)).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.)
            self.opt.step()
            return {'loss': float(loss.detach())}

    class Trainer:
        def __init__(self):
            self.wm = torch.nn.Linear(2, 1)
            self.policy = torch.nn.Linear(2, 1)
            self.critic = torch.nn.Linear(2, 1)
            self.wm_trainer = Update(self.wm)
            self.policy_trainer = Update(self.policy)
            self.policy_trainer.critic_opt = torch.optim.SGD(self.critic.parameters(), lr=.01)
            self.command = SimpleNamespace(bin_return=torch.zeros(2), bin_memory=torch.zeros(2))
            self.is_main = True
            self.log_dir = Path(folder)

        def act(self, obs):
            return self.policy(obs)

        def save(self, name='ckpt_latest.pt'):
            torch.save(trainer_state(self), self.log_dir / name)

    return Trainer()


class NumericalSafetyTests(unittest.TestCase):
    def test_guarded_valid_optimizer_update_is_identical_to_upstream_update(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(4)
            baseline = tiny_trainer(folder)
            torch.manual_seed(4)
            guarded = tiny_trainer(folder)
            baseline.wm_trainer.update()
            baseline.policy_trainer.update()
            with guard_training(guarded):
                guarded.wm_trainer.update()
                guarded.policy_trainer.update()
            for name in ('wm', 'policy', 'critic'):
                for left, right in zip(getattr(baseline, name).parameters(), getattr(guarded, name).parameters()):
                    self.assertTrue(torch.equal(left, right))

    def test_nonfinite_gradient_blocks_optimizer_step_and_restores_clip_function(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = tiny_trainer(folder)
            parameter = next(trainer.policy.parameters())
            before = parameter.detach().clone()
            original = torch.nn.utils.clip_grad_norm_
            with self.assertRaisesRegex(FloatingPointError, 'gradient norm'):
                with guard_training(trainer):
                    parameter.grad = torch.full_like(parameter, float('nan'))
                    torch.nn.utils.clip_grad_norm_(trainer.policy.parameters(), 1.)
                    trainer.policy_trainer.opt.step()
            self.assertTrue(torch.equal(parameter, before))
            self.assertIs(torch.nn.utils.clip_grad_norm_, original)

    def test_optimizer_hook_blocks_nonfinite_gradients_even_without_clipping(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = tiny_trainer(folder)
            parameter = next(trainer.policy.parameters())
            before = parameter.detach().clone()
            with guard_training(trainer):
                parameter.grad = torch.full_like(parameter, float('inf'))
                with self.assertRaisesRegex(FloatingPointError, 'policy gradients'):
                    trainer.policy_trainer.opt.step()
            self.assertTrue(torch.equal(parameter, before))

    def test_invalid_save_cannot_overwrite_the_last_finite_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = tiny_trainer(folder)
            path = Path(folder) / 'ckpt_latest.pt'
            with guard_training(trainer):
                trainer.save()
                saved = path.read_bytes()
                with torch.no_grad():
                    next(trainer.policy.parameters()).fill_(float('nan'))
                with self.assertRaisesRegex(FloatingPointError, 'checkpoint state.policy'):
                    trainer.save()
                self.assertEqual(saved, path.read_bytes())
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_checkpoint_loader_refuses_corrupted_network_or_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ckpt_latest.pt'
            data = dict(wm={'weight': torch.ones(1)}, policy={'weight': torch.ones(1)},
                        critic={'weight': torch.ones(1)}, config={}, iteration=1000)
            torch.save(data, path)
            self.assertEqual(load_checkpoint(path)['iteration'], 1000)
            data['policy_opt'] = {'state': {'value': torch.tensor(float('nan'))}}
            torch.save(data, path)
            with self.assertRaisesRegex(FloatingPointError, 'policy_opt'):
                load_checkpoint(path)

    def test_metrics_and_simulation_actions_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = tiny_trainer(folder)
            trainer.wm_trainer.update = lambda *args: {'loss': float('nan')}
            with guard_training(trainer):
                with self.assertRaisesRegex(FloatingPointError, 'world model update metrics'):
                    trainer.wm_trainer.update()
                with self.assertRaisesRegex(FloatingPointError, 'simulation actions'):
                    trainer.act(torch.full((1, 2), float('nan')))

    def test_finite_metrics_cannot_hide_a_corrupted_optimizer_state(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = tiny_trainer(folder)
            def update(*args):
                parameter = next(trainer.wm.parameters())
                trainer.wm_trainer.opt.state[parameter]['momentum_buffer'] = torch.full_like(parameter, float('inf'))
                return {'loss': 1.}
            trainer.wm_trainer.update = update
            with guard_training(trainer):
                with self.assertRaisesRegex(FloatingPointError, 'resulting state.wm_opt'):
                    trainer.wm_trainer.update()


if __name__ == '__main__':
    unittest.main()
