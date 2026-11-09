"""真实统计来源、BPTT 梯度、旧模型兼容和参考影子的功能回归。"""
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.normalization import RunningNorm
from g1_supermimic.variants import (
    TrainConfig, Optimizations, NormalizedWorldModel, history_frames, term_major,
    make_policy, make_critic, prepare_variant, residual_targets, orientation_goal,
)
from g1_supermimic.configuration import merge_config
from g1_supermimic.reference import ReferenceGhost
from g1_supermimic.core.policy.nets import Policy
from g1_supermimic.core.kinematics.rotations import quat_from_rotvec
from g1_supermimic.core.world_model.state_spec import StateSpec, StateSpecCfg


def tiny_config(enabled=True):
    """缩小网络宽度测试数值逻辑，不改变特征定义。"""
    cfg = TrainConfig()
    cfg.policy.encoder_hidden = cfg.policy.decoder_hidden = (16, 8)
    cfg.critic.hidden = (16,)
    cfg.world_model.hidden = (16, 16)
    if enabled:
        cfg.optimizations = Optimizations(True, True, True, True)
    prepare_variant(cfg)
    return cfg


class RunningNormTests(unittest.TestCase):
    def test_partitioned_statistics_match_population_moments(self):
        norm = RunningNorm(3)
        samples = torch.arange(90.).reshape(30, 3)
        for batch in samples.split([4, 7, 19]):
            norm.update(batch)
        torch.testing.assert_close(norm.mean, samples.double().mean(0))
        torch.testing.assert_close(norm.var, samples.double().var(0, unbiased=False))
        self.assertEqual(norm.count.item(), 30)

    def test_forward_is_frozen_and_preserves_gradients_and_units(self):
        norm = RunningNorm(3)
        norm.update(torch.randn(30, 3))
        before = {k: v.clone() for k, v in norm.state_dict().items()}
        x = torch.randn(4, 3, requires_grad=True)
        for mode in (True, False):
            norm.train(mode)
            torch.testing.assert_close(norm.denormalize(norm(x)), x)
        norm(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for k, v in norm.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_bad_samples_eval_and_limit_cannot_change_statistics(self):
        norm = RunningNorm(2, max_count=3)
        with self.assertRaises(FloatingPointError):
            norm.update(torch.tensor([[float('nan'), 1.]]))
        self.assertEqual(norm.count.item(), 0)
        norm.eval(); norm.update(torch.ones(3, 2))
        self.assertEqual(norm.count.item(), 0)
        norm.train(); norm.update(torch.ones(3, 2)); norm.update(torch.zeros(4, 2))
        self.assertEqual(norm.count.item(), 3)
        torch.testing.assert_close(norm.mean, torch.ones(2, dtype=torch.float64))


class VariantTests(unittest.TestCase):
    def test_optimization_flags_can_be_enabled_independently_for_ablations(self):
        for flag in vars(Optimizations()):
            with self.subTest(flag=flag):
                cfg = tiny_config(False)
                setattr(cfg.optimizations, flag, True)
                prepare_variant(cfg)
                policy, critic = make_policy(cfg), make_critic(cfg)
                goal = torch.randn(2, cfg.policy.goal_dim)
                self.assertEqual(policy(torch.randn(2, 930), goal).shape, (2, 29))
                self.assertEqual(critic(torch.randn(2, 71), goal, torch.randn(2, 33), torch.randn(2, 4)).shape, (2,))

    def test_legacy_config_restores_exact_original_policy_and_keys(self):
        cfg = tiny_config(False)
        original = Policy(cfg.policy)
        legacy = asdict(cfg); legacy.pop('optimizations')
        loaded_cfg = merge_config(TrainConfig(), legacy)
        restored = make_policy(loaded_cfg)
        restored.load_state_dict(original.state_dict(), strict=True)
        x, goal = torch.randn(2, 930), torch.randn(2, 680)
        torch.testing.assert_close(restored(x, goal), original(x, goal), rtol=0, atol=0)

    def test_history_layout_round_trip_and_latest_sample_count(self):
        frames = torch.arange(2 * 10 * 93.).reshape(2, 10, 93)
        flat = term_major(frames)
        torch.testing.assert_close(history_frames(flat, 10), frames)
        policy = make_policy(tiny_config())
        policy.update_normalization(flat, torch.randn(2, 689))
        self.assertEqual(policy.obs_norm.count.item(), 2)
        torch.testing.assert_close(policy.obs_norm.mean,
                                   history_frames(flat * policy.decoder.scale, 10)[:, -1].double().mean(0))

    def test_orientation_has_six_channels_without_position_leakage(self):
        cfg = tiny_config()
        policy, critic = make_policy(cfg), make_critic(cfg)
        obs, goal = torch.randn(2, 930), torch.randn(2, 689)
        changed = goal.clone(); changed[:, 680:683] += 1000
        feat, xi, error = torch.randn(2, 71), torch.randn(2, 33), torch.randn(2, 4)
        torch.testing.assert_close(policy(obs, goal), policy(obs, changed))
        torch.testing.assert_close(critic(feat, goal, xi, error), critic(feat, changed, xi, error))
        self.assertEqual(orientation_goal(goal, True).shape[-1], 686)
        # 朝向误差到动作的梯度存在；位置通道没有梯度。
        goal.requires_grad_(); policy(obs, goal).square().sum().backward()
        self.assertGreater(goal.grad[:, -6:].abs().sum().item(), 0)
        self.assertEqual(goal.grad[:, 680:683].abs().sum().item(), 0)

    def test_policy_critic_roundtrip_and_imagined_backward_do_not_update_moments(self):
        cfg = tiny_config()
        policy, critic = make_policy(cfg), make_critic(cfg)
        obs, goal = torch.randn(4, 930), torch.randn(4, 689)
        feat, xi, error = torch.randn(4, 71), torch.randn(4, 33), torch.randn(4, 4)
        policy.update_normalization(obs, goal)
        critic.update_normalization(feat, goal, xi, error)
        before = {k: v.clone() for k, v in policy.named_buffers()}
        x = obs.clone().requires_grad_()
        (policy(x, goal).square().sum() + critic(feat, goal, xi, error).sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for key, value in policy.named_buffers():
            torch.testing.assert_close(value, before[key])
        restored = make_policy(cfg)
        restored.load_state_dict(policy.state_dict(), strict=True)
        torch.testing.assert_close(restored(obs, goal), policy(obs, goal))

    def test_residual_targets_invert_explicit_and_nonexplicit_integration(self):
        for explicit in (True, False):
            spec = StateSpec(StateSpecCfg(explicit_integration=explicit), 'cpu')
            state = torch.randn(12, 71) * .1
            state[:, 3:7] = quat_from_rotvec(torch.randn(12, 3) * .3)
            delta = torch.randn(12, 70) * .1
            next_state = spec.integrate(state, delta)
            predicted = spec.integrate(state, residual_targets(spec, state, next_state))
            torch.testing.assert_close(predicted, next_state, atol=2.e-6, rtol=2.e-5)

    def test_world_model_updates_only_explicitly_and_keeps_action_gradient(self):
        cfg = tiny_config()
        world = NormalizedWorldModel(cfg.world_model, SimpleNamespace(), 'cpu')
        state = torch.randn(8, 71) * .1; state[:, 3:7] = quat_from_rotvec(torch.randn(8, 3) * .1)
        xi, action = torch.zeros(8, 33), torch.randn(8, 29)
        target = world.spec.integrate(state, torch.randn(8, 70) * .05)
        world.update_normalization(state, action, xi, target)
        before = {k: v.clone() for k, v in world.named_buffers()}
        # 原末层零初始化，先进行一次真实优化使动作梯度可观测。
        opt = torch.optim.Adam(world.parameters(), lr=.001)
        loss = world.spec.error_vector(world.step(state, action, xi), target).square().mean()
        loss.backward(); opt.step(); opt.zero_grad()
        action.requires_grad_(); world.step(state, action, xi).square().sum().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(action.grad.abs().sum().item(), 0)
        self.assertEqual(world.dynamics.input_norm.count.item(), 8)
        for key, value in world.named_buffers():
            torch.testing.assert_close(value, before[key])
        restored = NormalizedWorldModel(cfg.world_model, SimpleNamespace(), 'cpu')
        restored.load_state_dict(world.state_dict(), strict=True)
        torch.testing.assert_close(restored.step(state, action, xi), world.step(state, action, xi))

    def test_resume_rejects_structure_changes_but_accepts_matching_variant(self):
        cfg = tiny_config(False)
        previous = dict(config=asdict(cfg))
        cfg.optimizations.actor_input_normalization = True
        with self.assertRaisesRegex(ValueError, 'start a new run'):
            prepare_variant(cfg, previous)
        cfg = tiny_config()
        prepare_variant(cfg, dict(config=asdict(cfg)))


class ReferenceTests(unittest.TestCase):
    def test_ghost_uses_current_reference_world_pose_and_never_writes_simulator(self):
        model = SimpleNamespace(ngeom=2, qpos0=np.zeros(9), geom_contype=np.array([1, 0]),
                                geom_conaffinity=np.array([1, 0]), geom_rgba=np.ones((2, 4)))
        command = SimpleNamespace(num_envs=2, _env=SimpleNamespace(sim=SimpleNamespace(mj_model=model)),
            robot=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7), joint_q_adr=torch.tensor([7, 8]))),
            root_pos_w=torch.tensor([[5., 6., 7.], [8., 9., 10.]]),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2), joint_pos=torch.tensor([[.1, .2], [.3, .4]]))
        calls=[]
        vis=SimpleNamespace(get_env_indices=lambda n:[1], add_ghost_mesh=lambda qpos,**kw:calls.append((qpos,kw)))
        ghost=ReferenceGhost(command, alpha=.4, offset=(0, 1, 0))
        ghost.draw(vis)
        np.testing.assert_allclose(calls[-1][0][:3], [8, 10, 10])
        np.testing.assert_allclose(calls[-1][0][7:], [.3, .4])
        self.assertEqual(calls[-1][1]['alpha'], .4)
        self.assertEqual(ghost.model.geom_rgba[0,3], 0.)
        self.assertEqual(ghost.model.geom_rgba[1,3], .4)
        np.testing.assert_array_equal(model.geom_rgba, np.ones((2, 4)))
        command.joint_pos[1,0] = .8
        ghost.draw(vis)
        self.assertAlmostEqual(calls[-1][0][7], .8, places=6)
        ghost.enabled=False; ghost.draw(vis)
        self.assertEqual(len(calls),2)

    def test_invalid_visualization_parameters_are_rejected(self):
        for alpha in (0, -1, float('nan'), 2):
            with self.assertRaises(ValueError): ReferenceGhost(None, alpha=alpha)
        with self.assertRaises(ValueError): ReferenceGhost(None, offset=(0, float('inf'), 0))


if __name__ == '__main__':
    unittest.main()
