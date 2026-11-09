"""Cached-reference reward parity and differentiable rollout-cache behavior."""
from types import SimpleNamespace
import math
import unittest

import torch

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.reward_alignment import (
    body_rewards, CachedReferenceStep, bind_reward_alignment, alignment_manifest,
)
from g1_supermimic.variants import TrainConfig
from g1_supermimic.core.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms
from g1_supermimic.core.kinematics.rotations import quat_from_rotvec
from g1_supermimic.core.world_model.heads import ReferenceFrame


def reference(n=2):
    pos = torch.tensor([[[0., 0., 1.], [.2, 0., 1.]]]).repeat(n, 1, 1)
    quat = torch.tensor([[[1., 0., 0., 0.]]]).repeat(n, 2, 1)
    zeros = torch.zeros_like(pos)
    return ReferenceFrame(pos, quat, zeros, zeros, pos[:, 0], quat[:, 0])


class ToyHeads:
    anchor_index = 0
    reward_cfg = TrackingRewardCfg()

    def __init__(self):
        self.calls = 0

    def robot_bodies(self, state):
        self.calls += 1
        offset = torch.tensor([[0., 0., 0.], [.2, 0., 0.]])
        pos = state[:, None, :3] + offset
        quat = state[:, None, 3:7].expand(-1, 2, -1)
        return dict(pos=pos, quat=quat, lin_vel=torch.zeros_like(pos), ang_vel=torch.zeros_like(pos))

    def reward(self, state, action, prev_action, ref, bodies):
        return tracking_reward_terms(
            robot_body_pos_w=bodies['pos'], robot_body_quat_w=bodies['quat'],
            robot_body_lin_vel_w=bodies['lin_vel'], robot_body_ang_vel_w=bodies['ang_vel'],
            ref_body_pos_w=ref.body_pos_w, ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w, ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=0, action=action, prev_action=prev_action, joint_pos=state[:, 7:36],
            soft_limits=torch.tensor([[-1., 1.]]).repeat(29, 1), cfg=self.reward_cfg)


class ToyWorld:
    def __init__(self):
        self.heads = ToyHeads()

    def step(self, state, action, xi):
        # A minimal differentiable horizontal motion; all 71 state channels retain their meaning.
        return state + torch.cat([.1 * action[:, :1], torch.zeros_like(state[:, 1:])], -1)

    def termination_logits(self, state, ref):
        return torch.zeros(state.shape[0])


class RewardAlignmentTests(unittest.TestCase):
    def test_body_rewards_use_previous_anchor_and_preserve_actual_motion_penalty(self):
        ref = reference()
        # Co-located toy bodies isolate anchor translation from the rotation term.
        ref.body_pos_w[:, 1] = ref.body_pos_w[:, 0]
        yaw = quat_from_rotvec(torch.tensor([[0., 0., .3]]).repeat(2, 1))
        bodies = dict(pos=ref.body_pos_w + torch.tensor([.2, 0., 0.]),
                      quat=yaw[:, None].expand(-1, 2, -1))
        previous = torch.cat([ref.root_pos_w, ref.root_quat_w], -1)
        pos, ori = body_rewards(ToyHeads(), bodies, ref, previous)
        torch.testing.assert_close(pos, torch.full((2,), .02 * math.exp(-.2**2/.3**2)))
        torch.testing.assert_close(ori, torch.full((2,), .02 * math.exp(-.3**2/.4**2)))
        current = torch.cat([bodies['pos'][:, 0], bodies['quat'][:, 0]], -1)
        old_pos, old_ori = body_rewards(ToyHeads(), bodies, ref, current)
        torch.testing.assert_close(old_pos, torch.full((2,), .02))
        torch.testing.assert_close(old_ori, torch.full((2,), .02))

    def test_cached_fk_matches_recomputation_in_values_and_bptt_gradients(self):
        actions = (torch.randn(3, 2, 29) * .1).requires_grad_()
        def rollout(cached):
            world = ToyWorld()
            step = CachedReferenceStep(world, torch.zeros(29))
            state = torch.zeros(2, 71)
            state[:, 2], state[:, 3] = 1., 1.
            previous = torch.zeros(2, 29)
            rewards = []
            for action in actions:
                if not cached:
                    step.reset()
                output = step(state, action, previous, torch.zeros(2, 33), reference())
                state, reward = output[:2]
                rewards.append(reward)
                previous = action
            total = torch.stack(rewards)
            return total, torch.autograd.grad(total.sum(), actions)[0], world.heads.calls
        values, grad, calls = rollout(True)
        expected, expected_grad, uncached_calls = rollout(False)
        torch.testing.assert_close(values, expected, rtol=0, atol=0)
        torch.testing.assert_close(grad, expected_grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual((calls, uncached_calls), (4, 6))

    def test_fresh_rollout_never_reuses_a_previous_rollouts_anchor(self):
        step = CachedReferenceStep(ToyWorld(), torch.zeros(29))
        state = torch.zeros(2, 71); state[:, 2:4] = 1.
        step(state, torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 33), reference())
        other = state.clone(); other[:, 0] = 1.
        actual = step(other, torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 33), reference())
        step.reset()
        expected = step(other, torch.zeros(2, 29), torch.zeros(2, 29), torch.zeros(2, 33), reference())
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_update_cleanup_drops_autograd_cache_on_exception(self):
        cfg = TrainConfig(); cfg.model_alignment.cached_body_reference = True
        learner = SimpleNamespace()
        def fail():
            learner.step_fn._previous_state = torch.ones(1, requires_grad=True)
            raise RuntimeError('failed update')
        learner.update = fail
        trainer = SimpleNamespace(cfg=cfg, policy_trainer=learner, wm=ToyWorld(),
                                  lib=SimpleNamespace(default_joint_pos=torch.zeros(29)))
        bind_reward_alignment(trainer)
        with self.assertRaisesRegex(RuntimeError, 'failed update'):
            learner.update()
        self.assertIsNone(learner.step_fn._previous_state)
        self.assertIsNone(learner.step_fn._previous_anchor)

    def test_default_preserves_original_step_and_boolean_contract(self):
        cfg = TrainConfig(); original = object()
        learner = SimpleNamespace(step_fn=original)
        bind_reward_alignment(SimpleNamespace(cfg=cfg, policy_trainer=learner))
        self.assertIs(learner.step_fn, original)
        self.assertIsNone(alignment_manifest(cfg))
        cfg.model_alignment.cached_body_reference = 1
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            alignment_manifest(cfg)


if __name__ == '__main__':
    unittest.main()
