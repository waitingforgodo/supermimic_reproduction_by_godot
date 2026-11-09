"""Effective optimizer rates, resume state and the original equal-rate behavior."""
from types import SimpleNamespace
import unittest

import torch

from g1_supermimic.learning_rates import base_learning_rates, bind_learning_rates


def learner(actor=2e-4, critic=2e-4):
    params = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    optimizers = [torch.optim.Adam([p], lr=rate) for p, rate in
                  zip(params, (2e-4, actor, critic))]
    world, policy, value = optimizers
    for parameter, optimizer in zip(params, optimizers):
        parameter.square().backward()
        optimizer.step()
        optimizer.zero_grad()
    policy_trainer = SimpleNamespace(opt=policy, critic_opt=value)

    def upstream_set_lr(rate):
        for optimizer in (policy, value):
            for group in optimizer.param_groups:
                group['lr'] = rate
    policy_trainer.set_lr = upstream_set_lr
    cfg = SimpleNamespace(wm_trainer=SimpleNamespace(lr=2e-4),
                          policy_trainer=SimpleNamespace(lr=actor, critic_lr=critic))
    return SimpleNamespace(cfg=cfg, wm_trainer=SimpleNamespace(opt=world),
                           policy_trainer=policy_trainer), optimizers


class LearningRateTests(unittest.TestCase):
    def test_critic_follows_its_own_configured_cosine_scale(self):
        trainer, (world, actor, critic) = learner(critic=5e-5)
        bind_learning_rates(trainer)
        for factor in (1., .5, .03):
            world.param_groups[0]['lr'] = 2e-4 * factor
            trainer.policy_trainer.set_lr(2e-4 * factor)
            self.assertEqual(actor.param_groups[0]['lr'], 2e-4 * factor)
            self.assertAlmostEqual(critic.param_groups[0]['lr'], 5e-5 * factor)

    def test_equal_rates_preserve_upstream_values_exactly(self):
        trainer, (world, actor, critic) = learner()
        bind_learning_rates(trainer)
        for rate in (2e-4, .00013757309112146098, 6e-6):
            world.param_groups[0]['lr'] = rate
            expected = rate * 2e-4 / 2e-4
            trainer.policy_trainer.set_lr(expected)
            self.assertEqual(actor.param_groups[0]['lr'], expected)
            self.assertEqual(critic.param_groups[0]['lr'], expected)

    def test_frozen_actor_does_not_freeze_a_nonzero_critic(self):
        trainer, (world, actor, critic) = learner(actor=0., critic=1e-4)
        bind_learning_rates(trainer)
        world.param_groups[0]['lr'] = 1e-4
        trainer.policy_trainer.set_lr(0.)
        self.assertEqual(actor.param_groups[0]['lr'], 0.)
        self.assertAlmostEqual(critic.param_groups[0]['lr'], 5e-5)

    def test_rebinding_and_scheduling_keep_resumed_adam_moments(self):
        trainer, optimizers = learner(actor=1e-4, critic=1e-4)
        before = [{id(p): {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
                   for p, s in optimizer.state.items()} for optimizer in optimizers]
        bind_learning_rates(trainer)
        optimizers[0].param_groups[0]['lr'] = .00013757309112146098
        trainer.policy_trainer.set_lr(.00013757309112146098 * .5)
        for optimizer, saved in zip(optimizers, before):
            self.assertEqual({id(p) for p in optimizer.state}, set(saved))
            for parameter, state in optimizer.state.items():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, saved[id(parameter)][key], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, saved[id(parameter)][key])

    def test_invalid_rates_are_rejected_before_starting_training(self):
        for name, value in [('world_model', 0.), ('actor', -1.),
                            ('critic', float('nan')), ('critic', True)]:
            trainer, _ = learner()
            section = trainer.cfg.wm_trainer if name == 'world_model' else trainer.cfg.policy_trainer
            setattr(section, 'critic_lr' if name == 'critic' else 'lr', value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                base_learning_rates(trainer.cfg)


if __name__ == '__main__':
    unittest.main()
