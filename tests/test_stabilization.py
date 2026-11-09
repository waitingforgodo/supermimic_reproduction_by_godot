"""Actor constraint gradients, frozen references, backward compatibility and resume."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.configuration import digest
from g1_supermimic.stabilization import (
    prepare_stabilization, constrained_update, embed_reference_save, bind_stabilization,
)
from g1_supermimic.variants import TrainConfig, Optimizations, make_policy, prepare_variant


def tiny_config():
    cfg = TrainConfig()
    cfg.device = 'cpu'
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (8,)
    cfg.policy_trainer.compile = None
    cfg.optimizations = Optimizations(True, True, True, True)
    prepare_variant(cfg)
    return cfg


class ActorConstraintTests(unittest.TestCase):
    def run_update(self, horizon):
        actor = nn.Linear(1, 1, bias=False)
        reference = nn.Linear(1, 1, bias=False).eval().requires_grad_(False)
        with torch.no_grad():
            actor.weight.fill_(2.)
            reference.weight.fill_(1.)
        optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
        module = SimpleNamespace(policy_objective=lambda rollout, terminal, gamma:
                                 -horizon * rollout.first.square().mean())
        original = module.policy_objective
        def update():
            first = actor(torch.ones(2, 1))
            # Later imagined states must not become additional constraint samples.
            actor(torch.full((2, 1), 9.))
            rollout = SimpleNamespace(first=first, rewards=torch.zeros(horizon, 2))
            loss = -module.policy_objective(rollout, None, .99) / horizon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return {}
        metrics = constrained_update(update, actor, reference, .5, module)()
        self.assertIs(module.policy_objective, original)
        self.assertEqual(len(actor._forward_hooks), 0)
        self.assertIsNone(reference.weight.grad)
        self.assertEqual(reference.weight.item(), 1.)
        return actor.weight.item(), metrics

    def test_constraint_is_in_actor_loss_before_optimizer_and_independent_of_horizon(self):
        for horizon in (1, 16, 32):
            with self.subTest(horizon=horizon):
                weight, metrics = self.run_update(horizon)
                # d(w^2 + .5*(w-1)^2)/dw at w=2 is 5; SGD(.1) yields 1.5.
                self.assertAlmostEqual(weight, 1.5)
                self.assertAlmostEqual(metrics['reference_action_rms'], 1.)
                self.assertAlmostEqual(metrics['reference_penalty_loss'], .5)

    def test_exception_restores_objective_and_removes_forward_hook(self):
        actor = nn.Linear(1, 1)
        reference = nn.Linear(1, 1).eval().requires_grad_(False)
        original = lambda *a: torch.tensor(0.)
        module = SimpleNamespace(policy_objective=original)
        def update():
            actor(torch.ones(2, 1))
            raise RuntimeError('failed update')
        with self.assertRaisesRegex(RuntimeError, 'failed update'):
            constrained_update(update, actor, reference, 1., module)()
        self.assertIs(module.policy_objective, original)
        self.assertEqual(len(actor._forward_hooks), 0)


class ReferenceResumeTests(unittest.TestCase):
    def fixture(self, folder):
        cfg = tiny_config()
        source_cfg = asdict(cfg)
        reference = make_policy(cfg).eval()
        clip = Path(folder) / 'clip.npz'
        clip.write_bytes(b'reference data fixture')
        source = str(Path(folder) / 'removed_source.pt')
        payload = dict(config=source_cfg, policy=reference.state_dict(),
                       source_checkpoint=source, sha256='a' * 64, iteration=104000,
                       data_signature=[dict(name=clip.name, sha256=digest(clip))])
        cfg.resume = Path(folder) / 'resumed.pt'
        cfg.stability.reference_weight = 1.
        cfg.stability.reference_checkpoint = source
        return cfg, payload, clip

    def test_embedded_anchor_resumes_without_original_file_and_keeps_rng_and_buffers(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, payload, clip = self.fixture(folder)
            info = prepare_stabilization(cfg, dict(policy_reference=payload), [clip])
            self.assertEqual(info['reference_iteration'], 104000)
            self.assertEqual(cfg.stability.reference_sha256, 'a' * 64)
            actor = make_policy(cfg)
            learner = SimpleNamespace(update=lambda: {})
            trainer = SimpleNamespace(cfg=cfg, policy=actor, policy_trainer=learner, save=lambda *a: None)
            rng = torch.get_rng_state().clone()
            bind_stabilization(trainer)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            teacher = trainer.reference_policy
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            before = {k:v.clone() for k,v in teacher.named_buffers()}
            teacher(torch.randn(2, 930), torch.randn(2, 689))
            for key, value in teacher.named_buffers():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_changed_motion_or_pinned_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, payload, clip = self.fixture(folder)
            cfg.stability.reference_sha256 = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                prepare_stabilization(cfg, dict(policy_reference=payload), [clip])
            cfg.stability.reference_sha256 = None
            clip.write_bytes(b'changed reference data')
            with self.assertRaisesRegex(ValueError, 'motion hashes'):
                prepare_stabilization(cfg, dict(policy_reference=payload), [clip])

    def test_architecture_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, payload, clip = self.fixture(folder)
            cfg.policy.history = 11
            with self.assertRaisesRegex(ValueError, 'architecture'):
                prepare_stabilization(cfg, dict(policy_reference=payload), [clip])

    def test_save_embeds_frozen_reference_without_mutating_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg, payload, clip = self.fixture(folder)
            checkpoint = dict(policy=payload['policy'], config=asdict(cfg), iteration=105000)
            path = Path(folder) / 'candidate.pt'
            original = torch.save
            save = embed_reference_save(lambda: torch.save(checkpoint, path), payload)
            save()
            self.assertIs(torch.save, original)
            self.assertNotIn('policy_reference', checkpoint)
            loaded = torch.load(path, weights_only=False, map_location='cpu')
            self.assertEqual(loaded['policy_reference']['sha256'], payload['sha256'])
            for key, value in loaded['policy_reference']['policy'].items():
                torch.testing.assert_close(value, payload['policy'][key], rtol=0, atol=0)
            # The resulting checkpoint can supply the anchor after relocation.
            prepare_stabilization(cfg, loaded, [clip])

    def test_disabled_constraint_preserves_update_save_and_legacy_config(self):
        cfg = tiny_config()
        self.assertIsNone(prepare_stabilization(cfg, None, []))
        update, save = lambda: {}, lambda: None
        trainer = SimpleNamespace(cfg=cfg, policy_trainer=SimpleNamespace(update=update), save=save)
        bind_stabilization(trainer)
        self.assertIs(trainer.policy_trainer.update, update)
        self.assertIs(trainer.save, save)
        self.assertFalse(hasattr(trainer, 'reference_policy'))

    def test_invalid_weights_are_rejected_before_loading_reference(self):
        for weight in (-1., float('nan'), float('inf'), True, '1'):
            cfg = tiny_config()
            cfg.stability.reference_weight = weight
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                prepare_stabilization(cfg, None, [])


if __name__ == '__main__':
    unittest.main()
