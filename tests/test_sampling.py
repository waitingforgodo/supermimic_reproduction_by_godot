"""Analytic sampler tests: short tails are easy, falls forfeit future rewards."""
from types import SimpleNamespace as NS
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch

from g1_supermimic.sampling import (SamplingCfg, prepare_sampling, normalized_returns, bind_sampling,
                                          embed_sampler_save, VISITS_SATURATED, cap_per_clip, bind_clip_cap)


def normalize(rtg, lens, frames, terminated, ended, last, bootstrap=False):
    return normalized_returns(rtg, lens, frames, terminated, ended, last,
                              gamma=.99, max_step_reward=.1, g_max=10., bootstrapped=bootstrap)


class SamplingTests(unittest.TestCase):
    def test_perfect_short_tail_and_long_episode_get_same_score(self):
        for length in (1, 3, 100, 500):
            k = torch.arange(length, 0, -1, dtype=torch.float64)[None]
            achievable = .1 * (1 - .99 ** k) / (1 - .99)
            result = normalize(achievable, torch.tensor([length]),
                               torch.arange(length)[None], torch.tensor([False]),
                               torch.tensor([True]), torch.full((1, length), length-1))
            torch.testing.assert_close(result, torch.full_like(result, 10.), rtol=1e-11, atol=1e-11)

    def test_fall_does_not_receive_perfect_credit_for_short_survival(self):
        rtg = torch.tensor([[.1*(1+.99+.99**2), .1*(1+.99), .1]], dtype=torch.float64)
        result = normalize(rtg, torch.tensor([3]), torch.tensor([[20, 21, 22]]),
                           torch.tensor([True]), torch.tensor([False]), torch.full((1, 3), 99))
        expected = 10 * (1+.99+.99**2) / sum(.99**i for i in range(81))
        self.assertAlmostEqual(result[0, 0].item(), expected, places=12)
        self.assertLess(result.max(), 1.)

    def test_bootstrapped_timeout_keeps_critic_scale(self):
        value = torch.tensor([[7.3, 7.2]], dtype=torch.float64)
        result = normalize(value, torch.tensor([2]), torch.tensor([[100, 101]]),
                           torch.tensor([False]), torch.tensor([False]),
                           torch.full((1, 2), 999), bootstrap=True)
        torch.testing.assert_close(value, result)

    def test_enabling_resets_statistics_but_resuming_normalized_state_preserves_them(self):
        cfg = NS(sampling=SamplingCfg(True))
        report = prepare_sampling(cfg, {'config': {}})
        self.assertTrue(report['reset_sampler_state'])
        report = prepare_sampling(cfg, {'config': {'sampling': {'horizon_returns': True}}})
        self.assertFalse(report['reset_sampler_state'])
        cfg.sampling.horizon_returns = False
        with self.assertRaisesRegex(ValueError, 'normalized sampler'):
            prepare_sampling(cfg, {'config': {'sampling': {'horizon_returns': True}}})
        cfg.sampling.horizon_returns = 1
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            prepare_sampling(cfg)

    def trainer(self, enabled=True, reset=True):
        cmd = NS(cfg=NS(g_max=10., gamma=.99, ema_alpha=.05, min_remaining_steps=100),
                 library=NS(num_bins=2, last_frame_of=lambda frames: torch.full_like(frames, 2)),
                 bin_return=torch.tensor([2., 4.]), bin_memory=torch.tensor([.9, .1]),
                 ep_bins=torch.tensor([[0, 0, 1]]), ep_rewards=torch.tensor([[.1, .1, .1]]),
                 ep_len=torch.tensor([3]), clip_end=torch.tensor([True]),
                 _gamma_pow=.99**torch.arange(3, dtype=torch.float64), bootstrap_fn=None,
                 device='cpu', metrics={'episode_return_to_go': torch.zeros(1)},
                 _env=NS(termination_manager=NS(terminated=torch.tensor([False]))),
                 _log_step_rewards=lambda mask=None: None)
        return NS(cfg=NS(sampling=SamplingCfg(enabled), _sampling_reset_on_start=reset),
                  command=cmd, max_step_reward=.1, save=lambda name: None)

    def test_binding_uses_normalized_ema_and_clears_finished_episode(self):
        trainer = self.trainer()
        bind_sampling(trainer)
        command = trainer.command
        torch.testing.assert_close(command.bin_return, torch.full((2,), 10.))
        torch.testing.assert_close(command.bin_memory, torch.full((2,), .5))
        command.ep_frames[:] = torch.tensor([[0, 1, 2]])
        command._finalize_episodes(torch.tensor([0]))
        torch.testing.assert_close(command.bin_return, torch.full((2,), 10.))
        self.assertAlmostEqual(command.metrics['episode_return_to_go'][0].item(), 10., places=5)
        self.assertEqual(command.ep_len[0], 0)

    def test_disabled_binding_is_noop_and_enabled_resume_does_not_reset(self):
        trainer = self.trainer(enabled=False)
        before = trainer.command._log_step_rewards
        bind_sampling(trainer)
        self.assertIs(before, trainer.command._log_step_rewards)
        self.assertFalse(hasattr(trainer.command, 'ep_frames'))
        trainer = self.trainer(reset=False)
        bind_sampling(trainer)
        torch.testing.assert_close(trainer.command.bin_return, torch.tensor([2., 4.]))
        torch.testing.assert_close(trainer.command.bin_memory, torch.tensor([.9, .1]))

    def test_count_warmstart_learns_first_observation_then_converges_to_ema(self):
        trainer = self.trainer()
        trainer.cfg.sampling.count_warmstart = True
        bind_sampling(trainer)
        command = trainer.command
        command.ep_frames[:] = torch.tensor([[0, 1, 2]])
        command.ep_rewards.fill_(.05)
        command._finalize_episodes(torch.tensor([0]))
        torch.testing.assert_close(command.bin_return, torch.full((2,), 5.))
        torch.testing.assert_close(command.bin_visits, torch.ones(2, dtype=torch.int32))
        command.ep_len.fill_(3)
        command.ep_rewards.fill_(.1)
        command._finalize_episodes(torch.tensor([0]))
        torch.testing.assert_close(command.bin_return, torch.full((2,), 7.5))
        command.bin_visits.fill_(100)
        command.ep_len.fill_(3)
        command._finalize_episodes(torch.tensor([0]))
        torch.testing.assert_close(command.bin_return, torch.full((2,), 7.625))

    def test_visit_counts_roundtrip_and_legacy_fallback(self):
        trainer = self.trainer(reset=False)
        trainer.cfg._sampling_visits = torch.tensor([2, 17], dtype=torch.int32)
        bind_sampling(trainer)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            def save():
                torch.save(dict(policy={}, config={}, iteration=1), path)
            embed_sampler_save(save, trainer.command)()
            checkpoint = torch.load(path, weights_only=False)
            cfg = NS(sampling=SamplingCfg(True, True))
            checkpoint['config']['sampling'] = dict(horizon_returns=True, count_warmstart=True)
            report = prepare_sampling(cfg, checkpoint)
            self.assertEqual(report['visit_counter_initialization'], 'checkpoint')
            restored = self.trainer(reset=False)
            restored.cfg = cfg
            bind_sampling(restored)
            self.assertTrue(torch.equal(restored.command.bin_visits, trainer.command.bin_visits))
        legacy = self.trainer(reset=False)
        bind_sampling(legacy)
        self.assertTrue((legacy.command.bin_visits == VISITS_SATURATED).all())

    def test_invalid_counter_state_and_unscaled_warmstart_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires horizon_returns'):
            prepare_sampling(NS(sampling=SamplingCfg(False, True)))
        for visits in (torch.tensor([-1]), torch.tensor([1.]), torch.zeros(2, 2, dtype=torch.int32)):
            with self.subTest(visits=visits), self.assertRaisesRegex(ValueError, 'Invalid bin_visits'):
                prepare_sampling(NS(sampling=SamplingCfg(True)), dict(config={}, bin_visits=visits))

    def test_counter_save_composes_with_reference_policy_embedding(self):
        from g1_supermimic.stabilization import embed_reference_save
        command = NS(bin_visits=torch.tensor([3], dtype=torch.int32))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            def save():
                torch.save(dict(policy={}, config={}, iteration=1), path)
            wrapped = embed_reference_save(embed_sampler_save(save, command), dict(marker=17))
            wrapped()
            state = torch.load(path, weights_only=False)
            self.assertEqual(state['policy_reference']['marker'], 17)
            self.assertTrue(torch.equal(state['bin_visits'], command.bin_visits))

    def test_clip_cap_preserves_probability_and_within_clip_weights(self):
        hard = torch.tensor([.8, .1, .06, .04], dtype=torch.float64)
        groups = torch.tensor([0,0,1,2])
        capped = cap_per_clip(hard, groups, 3, .4)
        masses = torch.zeros(3, dtype=hard.dtype).index_add_(0, groups, capped)
        self.assertAlmostEqual(capped.sum().item(), 1., places=12)
        self.assertTrue((masses <= .40000001).all())
        self.assertAlmostEqual((capped[0]/capped[1]).item(), 8., places=12)
        self.assertIs(cap_per_clip(hard, groups, 3, 0.), hard)

    def test_clip_cap_handles_zero_softmax_mass_and_feasibility(self):
        groups = torch.arange(4)
        hard = torch.tensor([1., 0., 0., 0.])
        capped = cap_per_clip(hard, groups, 4, .4)
        torch.testing.assert_close(capped, torch.tensor([.4,.2,.2,.2]))
        uniform = cap_per_clip(hard, groups, 4, .25)
        torch.testing.assert_close(uniform, torch.full((4,), .25))
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            cap_per_clip(hard, groups, 4, .1)
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            prepare_sampling(NS(sampling=SamplingCfg(True, True, .1)), num_clips=1)
        self.assertEqual(prepare_sampling(NS(sampling=SamplingCfg(True, True, .1)), num_clips=40)['hard_clip_cap'], .1)
        for cap in (-.1, 1.1, True, float('nan')):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                prepare_sampling(NS(sampling=SamplingCfg(True, True, cap)))

    def test_live_reset_sampler_uses_capped_distribution_and_preserves_other_modes(self):
        lib=NS(num_clips=2,num_bins=4,bin_clip=torch.tensor([0,0,1,1]),
               bin_start=torch.tensor([0,10,20,30]),bin_end=torch.tensor([10,20,30,40]),
               last_frame_of=lambda f:torch.where(f<20,19,39),
               clip_start=torch.tensor([0,20]),frame_clip=torch.tensor([0]*20+[1]*20))
        cfg=NS(sampling_mode='return',softmax_temp=25.,g_max=10.,memory_alpha=.01,
               uniform_ratio=0.,hard_ratio=1.,memory_ratio=0.)
        command=NS(cfg=cfg,library=lib,device='cpu',bin_return=torch.tensor([0.,10.,10.,10.]),
                   bin_memory=torch.full((4,),.25),metrics={'sampling_entropy':torch.zeros(1)},
                   _sample_frames=lambda n:torch.full((n,),-123))
        bind_clip_cap(command,.6)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            frames=command._sample_frames(20000)
        self.assertTrue((frames>=0).all())
        self.assertTrue((frames<=37).all())
        self.assertAlmostEqual((frames<20).float().mean().item(),.6,delta=.02)
        cfg.sampling_mode='start'
        self.assertTrue((command._sample_frames(3)==-123).all())


if __name__ == '__main__':
    unittest.main()
