"""New-motion initialization must not resume old data, optimizers or scores."""
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch
from torch import nn

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.cli import parser
from g1_supermimic.configuration import digest
from g1_supermimic.initialization import apply_initialization
from g1_supermimic.training import configure
from g1_supermimic.retention import retain_best
from g1_supermimic.ema import bind_ema, PolicyEMACfg
import test_reproduced as reproduced
import test_retention as retention
from test_variants import tiny_config


def fixture(root):
    source, target = root/'source', root/'target'
    source.mkdir(); target.mkdir()
    old = reproduced.ReproducedTests().motion_fixture(source)
    new = reproduced.ReproducedTests().motion_fixture(target)
    with np.load(new) as data:
        values = dict(data)
    values['joint_pos'] = values['joint_pos'] + .01
    np.savez(new, **values)
    cfg = tiny_config()
    cfg.data.npz_dir, cfg.data.clips = source, ['clip']
    cfg.env.num_envs = 17
    cfg.policy_trainer.horizon, cfg.policy_trainer.batch_size = 32, 7
    cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    cfg.sampling.horizon_returns = cfg.sampling.count_warmstart = True
    checkpoint = source/'best.pt'
    torch.save(dict(config=asdict(cfg), iteration=125000, num_updates=49888,
                    wm={}, policy={}, critic={}, bin_visits=torch.tensor([99]),
                    bin_return=torch.tensor([-5.]), policy_opt={'old': True}), checkpoint)
    (source/'run_manifest.json').write_text(json.dumps(dict(implementation='Godot-SuperMimic',
        data=[dict(path=str(old), sha256=digest(old))])))
    return checkpoint, new


class InitializationTests(unittest.TestCase):
    def test_new_motions_inherit_architecture_but_not_resume_state(self):
        with TemporaryDirectory() as folder:
            source, motion = fixture(Path(folder))
            before = digest(source)
            args = parser().parse_args(['train', '--init-from', str(source),
                '--motion', str(motion), '--iterations', '20000'])
            cfg, manifest = configure(args)
            self.assertIsNone(cfg.resume)
            self.assertEqual(cfg.env.num_envs, 17)
            self.assertEqual((cfg.policy_trainer.horizon, cfg.policy_trainer.batch_size), (32, 7))
            self.assertTrue(cfg._sampling_reset_on_start)
            self.assertIsNone(cfg._sampling_visits)
            self.assertEqual(set(cfg._initial_weights), {'wm', 'policy', 'critic'})
            self.assertEqual(manifest['configuration_base'], 'checkpoint_weights')
            self.assertEqual(manifest['initialization']['source_iteration'], 125000)
            self.assertEqual(manifest['initialization']['source_sha256'], before)
            self.assertEqual(manifest['data'][0]['sha256'], digest(motion))
            self.assertEqual(digest(source), before)
            # Normal resume still rejects this exact data change.
            args = parser().parse_args(['resume', '--checkpoint', str(source),
                '--motion', str(motion), '--iterations', '320000'])
            with self.assertRaisesRegex(ValueError, 'hashes/order'):
                configure(args)

    def test_initialization_uses_evaluated_ema_and_starts_new_ema_history(self):
        with TemporaryDirectory() as folder:
            source, motion = fixture(Path(folder))
            ck = torch.load(source, weights_only=False)
            ck['config']['ema'] = {'decay': .995}
            ck.update(policy={'weight': torch.tensor(1.)},
                      policy_ema={'weight': torch.tensor(2.)}, ema_updates=123)
            torch.save(ck, source)
            cfg, manifest = configure(parser().parse_args(['train', '--init-from', str(source), '--motion', str(motion)]))
            self.assertEqual(cfg._initial_weights['policy']['weight'], 2)
            self.assertEqual(cfg._ema_updates, 0)
            self.assertIsNone(cfg._ema_resume)
            self.assertEqual(manifest['initialization']['policy_source'], 'ema')

    def test_missing_target_and_changed_architecture_fail_before_simulation(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, motion = fixture(root)
            with self.assertRaisesRegex(ValueError, 'explicit target'):
                configure(parser().parse_args(['train', '--init-from', str(source)]))
            override = root/'override.json'
            override.write_text(json.dumps({'policy': {'encoder_hidden': [32, 8]}}))
            with self.assertRaisesRegex(ValueError, 'same policy architecture'):
                configure(parser().parse_args(['train', '--init-from', str(source),
                    '--motion', str(motion), '--config', str(override)]))
            with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                configure(parser().parse_args(['resume', '--checkpoint', str(source),
                    '--init-from', str(source), '--motion', str(motion)]))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['train', '--resume', 'a.pt', '--init-from', 'b.pt'])

    def test_weights_and_buffers_load_into_fresh_optimizers_before_ema(self):
        modules = {name: nn.Linear(2, 2) for name in ('wm', 'policy', 'critic')}
        for module in modules.values():
            module.register_buffer('calibration', torch.tensor(0.))
        source = deepcopy(modules)
        with torch.no_grad():
            for module in source.values():
                for p in module.parameters(): p.fill_(3.)
                module.calibration.fill_(7.)
        opts = [torch.optim.Adam(m.parameters()) for m in modules.values()]
        cfg = NS(_initial_weights={k:v.state_dict() for k,v in source.items()},
                 _init_from='parent.pt', ema=PolicyEMACfg(.9), resume=None)
        tr = NS(**modules, cfg=cfg, is_main=False, iteration=0, num_updates=0,
                wm_trainer=NS(opt=opts[0]),
                policy_trainer=NS(opt=opts[1], critic_opt=opts[2]), save=lambda *a:None)
        apply_initialization(tr)
        bind_ema(tr)
        self.assertFalse(hasattr(cfg, '_initial_weights'))
        self.assertEqual((tr.iteration, tr.num_updates, tr.policy_ema.updates), (0, 0, 0))
        self.assertTrue(all(not opt.state for opt in opts))
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, source[name].state_dict()[key], rtol=0, atol=0)
        for key, value in tr.policy_ema.state_dict().items():
            torch.testing.assert_close(value, source['policy'].state_dict()[key], rtol=0, atol=0)

    def test_target_baseline_is_evaluated_at_zero_and_protected_from_regression(self):
        with TemporaryDirectory() as folder:
            tr = retention.trainer(folder)
            tr.iteration = tr.num_updates = 0
            tr.cfg._init_from = 'old_dataset_parent.pt'
            tr.cfg.checkpoint_confirmations = 2
            calls = []
            def evaluate(*args):
                calls.append(tr.iteration)
                return retention.report(failed=(False,False), steps=(101,101),
                                        reward=.7 if tr.iteration==0 else .6)
            with retain_best(tr, dict(data=retention.DATA), evaluator=evaluate):
                self.assertEqual(calls, [0, 0])
                best = Path(folder)/'best.pt'
                before = best.read_bytes()
                ck = torch.load(best, weights_only=False)
                self.assertEqual(ck['iteration'], 0)
                self.assertEqual(ck['selection']['report']['data'], retention.DATA)
                tr.save()  # No duplicate evaluation at the same counters.
                tr.iteration, tr.num_updates = 64, 10
                tr.save()
                self.assertEqual(best.read_bytes(), before)
            self.assertEqual(calls, [0, 0, 64])


if __name__ == '__main__':
    unittest.main()
