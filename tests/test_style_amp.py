"""Adversarial BPTT gradients, real history boundaries and checkpoint isolation."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import torch
from torch import nn

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.core.data.replay_buffer import ReplayBuffer
from g1_supermimic.core.kinematics.rotations import quat_from_rotvec, quat_apply, quat_apply_inverse, quat_mul
from g1_supermimic.style_amp import (
    StyleAMPCfg, StyleAMP, bind_style_amp, complete_window_mask,
    frame_features, prepare_style_amp, real_windows,
)
from g1_supermimic.ema import PolicyEMACfg, bind_ema
from g1_supermimic.safety import guard_training
from test_safety import tiny_trainer
from test_variants import tiny_config


class Library:
    def __init__(self):
        self.num_frames = 48
        self.default_joint_pos = torch.zeros(29)
        self.frame_clip = torch.arange(2).repeat_interleave(24)
        self.states = torch.zeros(48, 71)
        self.states[:, 3:7] = quat_from_rotvec(torch.stack((torch.zeros(48), torch.zeros(48), torch.arange(48)*.03), -1))
        self.states[:, 7:36] = torch.sin(torch.arange(48)[:, None]*.1+torch.arange(29)[None, :]*.05)*.2
        self.states[:, 36] = .3
        self.states[:, 39:42] = torch.tensor([.1, .2, .3])  # NPZ convention: WORLD angular velocity
        self.states[:, 42:71] = .1

    def ref71(self, indices):
        return self.states[indices]


def buffer_fixture():
    b = ReplayBuffer(2, 32, 71, 29, 33, 'cpu')
    ids = torch.arange(2)
    for t in range(90):
        terminal = (t+ids)%19 == 18
        state = torch.zeros(2, 71)
        state[:, 0] = ids*1000+t
        state[:, 3] = 1
        state[:, 7:36] = .01*t
        b.add(ids, state, torch.zeros(2,29), torch.zeros(2,29), torch.zeros(2,33),
              torch.zeros(2), terminal.float(), torch.full((2,),t), torch.zeros(2,dtype=torch.long),
              terminal, (t+ids)%19==0)
        b.end_episode(ids[terminal])
    return b


def options(**kwargs):
    return replace(StyleAMPCfg(weight=.25, hidden=(8,), batch_size=4), **kwargs)


class StyleAMPTests(unittest.TestCase):
    def test_features_ignore_world_translation_and_heading(self):
        torch.manual_seed(2)
        state = torch.randn(5,71)*.1
        state[:, 3:7] = quat_from_rotvec(torch.randn(5,3)*.2)
        rotated = state.clone()
        yaw = quat_from_rotvec(torch.tensor([[0.,0.,1.1]]).expand(5,-1))
        rotated[:, :3] = quat_apply(yaw,state[:, :3])+100
        rotated[:, 3:7] = quat_mul(yaw,state[:, 3:7])
        rotated[:, 36:39] = quat_apply(yaw,state[:, 36:39])
        torch.testing.assert_close(frame_features(state,torch.zeros(29)),
                                   frame_features(rotated,torch.zeros(29)),atol=1.e-6,rtol=1.e-6)

    def test_experts_use_body_angular_velocity_and_never_cross_clips(self):
        lib = Library(); style = StyleAMP(options(),lib,[['motions','hash']])
        self.assertEqual(style._expert_starts.numel(),30)
        ids = style._expert_starts
        self.assertTrue((lib.frame_clip[ids]==lib.frame_clip[ids+9]).all())
        expected = quat_apply_inverse(lib.states[:,3:7],lib.states[:,39:42])
        torch.testing.assert_close(style._experts[:,-3:],expected)
        self.assertEqual(style.expert_windows(7).shape,(7,10,67))

    def test_real_windows_exclude_terminals_and_cuts_after_ring_wrap(self):
        b = buffer_fixture()
        window = real_windows(b,10,512)
        self.assertEqual(window.shape,(512,10,71))
        torch.testing.assert_close(window[:,1:,0]-window[:,:-1,0],torch.ones(512,9))
        e,t = b._sample_from_mask(complete_window_mask(b,10),512)
        for k in range(10):
            row=(t+k)%b.T
            self.assertTrue((b.ep_id[e,row]==b.ep_id[e,t]).all())
            self.assertFalse(b.terminal[e,row].any())

    def test_quadratic_reward_is_bounded_with_exact_reference_values(self):
        style = StyleAMP(options(),Library(),[])
        windows = torch.randn(3,10,67)
        for score,value in ((-2.,0.),(-1.,0.),(0.,.75),(1.,1.),(3.,0.)):
            with torch.no_grad():
                style.discriminator[-1].weight.zero_();style.discriminator[-1].bias.fill_(score)
            torch.testing.assert_close(style.style_reward(windows),torch.full((3,),value))

    def test_frozen_style_preserves_bptt_through_frozen_dynamics(self):
        style = StyleAMP(options(),Library(),[])
        with torch.no_grad():
            style.discriminator[0].weight.fill_(.001);style.discriminator[0].bias.fill_(1)
            style.discriminator[-1].weight.fill_(.01);style.discriminator[-1].bias.zero_()
        actor = nn.Linear(2,2)
        dynamics = nn.Linear(2,2).requires_grad_(False)
        x = torch.ones(3,2)*.1
        def state(value):
            prefix = torch.zeros(len(value),7);prefix[:,3]=1
            return torch.cat((prefix,value,torch.zeros(len(value),62)),1)
        states = [state(x)]
        for _ in range(4):
            x=x+.01*dynamics(actor(x))
            states.append(state(x))
        start = NS(hist_states=states[0].detach()[None].expand(10,-1,-1))
        base = torch.zeros(4,3)
        ro = NS(states=torch.stack(states),rewards=base.clone(),reward_terms={})
        before = deepcopy(style.state_dict())
        with style.frozen():
            result = style.add_rollout_reward(ro,start,.02)
            self.assertEqual(result.rewards.shape,(4,3))
            self.assertTrue((result.rewards.detach()>=0).all())
            self.assertTrue((result.rewards.detach()<=.25*.02).all())
            (-result.rewards.mean()).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.parameters()),0)
        self.assertTrue(all(p.grad is None for p in dynamics.parameters()))
        self.assertTrue(all(p.grad is None for p in style.discriminator.parameters()))
        for key,value in style.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_discriminator_adam_continues_exactly_after_restore(self):
        torch.manual_seed(5)
        lib = Library();b=buffer_fixture()
        a=StyleAMP(options(),lib,[['x','hash']])
        a.update(b)
        saved=deepcopy(a.checkpoint())
        c=StyleAMP(options(),lib,[['x','hash']],saved)
        torch.manual_seed(77);a.update(b)
        torch.manual_seed(77);c.update(b)
        self.assertEqual((a.updates,c.updates),(2,2))
        for key,value in a.state_dict().items():torch.testing.assert_close(value,c.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in a.discriminator.parameters()))
        bad=deepcopy(saved);bad['expert_signature']=[['different','hash']]
        with self.assertRaisesRegex(ValueError,'does not match'):StyleAMP(options(),lib,[['x','hash']],bad)
        bad=deepcopy(saved);bad['model']['std'].zero_()
        with self.assertRaisesRegex(ValueError,'standard deviation'):StyleAMP(options(),lib,[['x','hash']],bad)

    def test_bad_config_or_missing_resume_state_rejected(self):
        cfg=tiny_config();cfg.wm_trainer.compile=cfg.policy_trainer.compile=None
        cfg.style_amp=options()
        with TemporaryDirectory() as folder:
            file=Path(folder)/'x.npz';file.write_bytes(b'fixture')
            report=prepare_style_amp(cfg,None,[file])
            self.assertEqual(report['physical_reward_and_best_selection'],'unchanged tracking only')
            with self.assertRaisesRegex(ValueError,'missing discriminator'):
                prepare_style_amp(cfg,dict(config={'style_amp':{'weight':.25}}),[file])
            for change in ({'weight':True},{'gradient_penalty':float('nan')},{'history':11},{'updates_per_policy':0}):
                cfg.style_amp=options(**change)
                with self.assertRaises(ValueError):prepare_style_amp(cfg,None,[file])
            cfg.style_amp=options();cfg.performance.precision='bf16'
            with self.assertRaisesRegex(ValueError,'FP32 eager'):prepare_style_amp(cfg,None,[file])

    def test_nonfinite_discriminator_step_is_blocked_before_optimizer(self):
        style=StyleAMP(options(),Library(),[])
        before=deepcopy(style.state_dict())
        with TemporaryDirectory() as folder:
            tr=tiny_trainer(folder);tr.style_amp=style
            with guard_training(tr):
                for p in style.discriminator.parameters():p.grad=torch.full_like(p,float('nan'))
                with self.assertRaisesRegex(FloatingPointError,'AMP discriminator gradients'):style.opt.step()
        self.assertFalse(style.opt.state)
        for key,value in style.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_save_composes_with_ema_and_only_saves_persistent_style_state(self):
        with TemporaryDirectory() as folder:
            tr=tiny_trainer(folder);tr.device='cpu';tr.lib=Library()
            tr.cfg=NS(style_amp=options(),ema=PolicyEMACfg(.9),resume=None,seed=3,_style_signature=[])
            bind_ema(tr);bind_style_amp(tr);tr.save()
            ck=torch.load(Path(folder)/'ckpt_latest.pt',weights_only=False)
            self.assertIn('policy_ema',ck)
            self.assertEqual(ck['style_amp']['updates'],0)
            self.assertIn('optimizer',ck['style_amp'])
            self.assertFalse(any(key.startswith('_expert') for key in ck['style_amp']['model']))


if __name__=='__main__':unittest.main()
