"""EMA resume/selection, replay boundary semantics and mixed-precision BPTT."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from g1_supermimic.runtime import activate
activate()
from g1_supermimic.core.data.replay_buffer import ReplayBuffer
from g1_supermimic.ema import PolicyEMA, PolicyEMACfg, bind_ema, policy_state, prepare_ema
from g1_supermimic.performance import PerformanceCfg, autocast_mlp, autocast_network, autocast_region, bind_replay_cache, prepare_performance
from g1_supermimic.normalization import RunningNorm
from g1_supermimic.safety import guard_training,require_finite
from test_safety import tiny_trainer
from test_retention import DATA, report, trainer as retention_trainer
from g1_supermimic.retention import retain_best


class AveragingTests(unittest.TestCase):
    def test_adam_and_average_continue_identically_after_checkpoint_roundtrip(self):
        torch.manual_seed(3)
        p=nn.Linear(3,2);opt=torch.optim.Adam(p.parameters(),lr=.01)
        avg=PolicyEMA(p,.9)
        opt.register_step_post_hook(lambda *unused:avg.update())
        def step(model,optimizer):
            optimizer.zero_grad();model(torch.ones(4,3)).square().mean().backward();optimizer.step()
        for _ in range(3):step(p,opt)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt'
            torch.save(dict(policy=p.state_dict(),policy_opt=opt.state_dict(),policy_ema=avg.state_dict(),ema_updates=avg.updates),path)
            ck=torch.load(path,weights_only=False)
        q=nn.Linear(3,2);q.load_state_dict(ck['policy'])
        qo=torch.optim.Adam(q.parameters(),lr=.01);qo.load_state_dict(ck['policy_opt'])
        qa=PolicyEMA(q,.9,ck['policy_ema'],ck['ema_updates'])
        qo.register_step_post_hook(lambda *unused:qa.update())
        for _ in range(2):step(p,opt);step(q,qo)
        for k,v in p.state_dict().items():torch.testing.assert_close(v,q.state_dict()[k],rtol=0,atol=0)
        for k,v in avg.state_dict().items():torch.testing.assert_close(v,qa.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(qa.updates,5)

    def test_formula_copies_buffers_preserves_online_and_resumes_exactly(self):
        online = nn.Sequential(nn.Linear(2, 2), RunningNorm(2, max_count=4))
        average = PolicyEMA(online, .9)
        before = {k: v.clone() for k, v in online.named_parameters()}
        with torch.no_grad():
            for p in online.parameters(): p.add_(1)
        online[1].update(torch.randn(4, 2))
        average.update()
        for name, p in average.module.named_parameters():
            torch.testing.assert_close(p, before[name] + .1)
            self.assertFalse(p.requires_grad)
        for name, value in online.named_buffers():
            torch.testing.assert_close(dict(average.module.named_buffers())[name], value)
        resumed = PolicyEMA(online, .9, deepcopy(average.state_dict()), average.updates)
        for _ in range(3):
            with torch.no_grad():
                for p in online.parameters(): p.mul_(.8)
            average.update(); resumed.update()
        for k, v in average.state_dict().items():
            torch.testing.assert_close(v, resumed.state_dict()[k], rtol=0, atol=0)
        self.assertEqual(resumed.updates, 4)

    def test_save_keeps_online_optimizer_and_ema_separate(self):
        with TemporaryDirectory() as folder:
            tr = tiny_trainer(folder)
            tr.cfg = NS(ema=PolicyEMACfg(.9),resume=None)
            parent = {k:v.clone() for k,v in tr.policy.state_dict().items()}
            bind_ema(tr)
            with guard_training(tr):
                tr.policy_trainer.update()
                tr.save()
            ck = torch.load(Path(folder)/'ckpt_latest.pt', weights_only=False)
            self.assertEqual(ck['ema_updates'], 1)
            for k,v in tr.policy.state_dict().items():
                torch.testing.assert_close(ck['policy'][k], v)
                torch.testing.assert_close(ck['policy_ema'][k], parent[k]*.9+v*.1)
            self.assertIn('policy_opt', ck)

    def test_blocked_nonfinite_step_does_not_advance_average(self):
        with TemporaryDirectory() as folder:
            tr = tiny_trainer(folder); tr.cfg = NS(ema=PolicyEMACfg(.9),resume=None)
            bind_ema(tr)
            before = deepcopy(tr.policy_ema.state_dict())
            with guard_training(tr):
                for p in tr.policy.parameters(): p.grad = torch.full_like(p, float('nan'))
                with self.assertRaises(FloatingPointError): tr.policy_trainer.opt.step()
            self.assertEqual(tr.policy_ema.updates, 0)
            for k,v in before.items(): torch.testing.assert_close(v, tr.policy_ema.state_dict()[k])

    def test_policy_choice_is_explicit_and_old_checkpoints_are_online(self):
        ck = dict(config={}, policy={'x':torch.tensor(1)})
        self.assertEqual(policy_state(ck)[1], 'online')
        with self.assertRaisesRegex(ValueError,'no EMA'): policy_state(ck,'ema')
        ck.update(config={'ema':{'decay':.995}},policy_ema={'x':torch.tensor(2)})
        self.assertEqual(policy_state(ck)[0]['x'], 2)
        self.assertEqual(policy_state(ck,'online')[0]['x'], 1)
        del ck['policy_ema']
        with self.assertRaisesRegex(ValueError,'no EMA'): policy_state(ck)

    def test_save_composes_with_sampler_and_reference_wrappers(self):
        from g1_supermimic.sampling import embed_sampler_save
        from g1_supermimic.stabilization import embed_reference_save
        from g1_supermimic.safety import trainer_state
        with TemporaryDirectory() as folder:
            tr=tiny_trainer(folder);tr.cfg=NS(ema=PolicyEMACfg(.9),resume=None)
            tr.command.bin_visits=torch.tensor([2,7],dtype=torch.int32)
            def save(name='ckpt_latest.pt'):
                torch.save(dict(trainer_state(tr),config={},iteration=1),Path(folder)/name)
            tr.save=embed_reference_save(embed_sampler_save(save,tr.command),{'source':'parent'})
            bind_ema(tr);tr.save()
            ck=torch.load(Path(folder)/'ckpt_latest.pt',weights_only=False)
            self.assertEqual(ck['policy_reference']['source'],'parent')
            torch.testing.assert_close(ck['bin_visits'],tr.command.bin_visits)
            self.assertIn('policy_ema',ck)

    def test_configuration_validates_decay_and_preserves_resume_average(self):
        for decay in (True,-.1,1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): prepare_ema(NS(ema=PolicyEMACfg(decay)))
        cfg=NS(ema=PolicyEMACfg(.9))
        saved=dict(config={'ema':{'decay':.9}},policy_ema={'w':torch.ones(1)},ema_updates=7)
        self.assertEqual(prepare_ema(cfg,saved)['initialized_from'],'saved EMA')
        self.assertEqual(cfg._ema_updates,7)
        with self.assertRaisesRegex(ValueError,'Cannot disable'):
            prepare_ema(NS(ema=PolicyEMACfg()),saved)
        del saved['policy_ema']
        with self.assertRaisesRegex(ValueError,'missing'): prepare_ema(cfg,saved)

    def test_retention_refuses_online_score_for_ema_and_keeps_both_states(self):
        with TemporaryDirectory() as folder:
            tr=retention_trainer(folder);tr.cfg.ema=PolicyEMACfg(.9)
            with self.assertRaisesRegex(ValueError,'wrong policy source'):
                with retain_best(tr,dict(data=DATA),evaluator=lambda *a:report()):tr.save()
            self.assertFalse((Path(folder)/'best.pt').exists())
        with TemporaryDirectory() as folder:
            tr=retention_trainer(folder);tr.cfg.ema=PolicyEMACfg(.9)
            # The integration's save wrapper includes both online and EMA tensors.
            original=tr.save
            def save(name='ckpt_latest.pt'):
                original(name)
                p=Path(folder)/name; ck=torch.load(p,weights_only=False)
                ck.update(config={'ema':{'decay':.9}},policy_ema={'weight':torch.zeros(1)})
                torch.save(ck,p)
            tr.save=save
            def evaluate(*unused):
                value=report();value['policy_source']='ema';return value
            with retain_best(tr,dict(data=DATA),evaluator=evaluate):tr.save()
            ck=torch.load(Path(folder)/'best.pt',weights_only=False)
            self.assertEqual(ck['selection']['report']['policy_source'],'ema')
            self.assertEqual(policy_state(ck)[0]['weight'],0)
            self.assertEqual(ck['policy']['weight'],1)


def populate(buffer, steps=45):
    E=buffer.E;ids=torch.arange(E)
    for t in range(steps):
        terminal=(t+ids)%7==6
        state=torch.zeros(E,71);state[:,0]=ids*1000+t
        buffer.add(ids,state,torch.zeros(E,29),torch.zeros(E,29),torch.zeros(E,33),
                   torch.zeros(E),terminal.float(),torch.full((E,),t),torch.zeros(E,dtype=torch.long),
                   terminal,(t+ids)%7==0)
        buffer.end_episode(ids[terminal])


class ReplayTests(unittest.TestCase):
    def test_windows_and_history_match_reference_after_wrap_and_episode_cuts(self):
        ref=ReplayBuffer(3,32,71,29,33,'cpu');populate(ref)
        fast=deepcopy(ref);bind_replay_cache(fast)
        for k in (1,3,8):
            torch.testing.assert_close(fast.window_mask(k),ref.window_mask(k))
            mask=ref.window_mask(k)
            if not mask.any():continue
            e,t=fast._sample_from_mask(fast.window_mask(k),1000)
            self.assertTrue(mask[e,t].all())
            window=fast.sample_windows(100,k)
            self.assertTrue(torch.equal(window.states[1:,:,0]-window.states[:-1,:,0],torch.ones(k,100)))
        start=fast.sample_starts(100,10)
        # Backfilled histories repeat the first state, then advance within one episode.
        diff=start.hist_states[1:,:,0]-start.hist_states[:-1,:,0]
        self.assertTrue(((diff==0)|(diff==1)).all())
        torch.testing.assert_close(start.hist_states[-1],start.state)

    def test_cached_masks_and_indices_are_invalidated_before_every_write(self):
        b=ReplayBuffer(2,32,71,29,33,'cpu');populate(b,25);bind_replay_cache(b)
        mask=b.window_mask(2)
        for _ in range(10): b.sample_windows(8,2)
        self.assertIs(b.window_mask(2),mask)
        self.assertEqual(b._replay_cache_stats['mask_builds'],1)
        self.assertEqual(b._replay_cache_stats['index_builds'],1)
        self.assertEqual(b._replay_cache_stats['cache_hits'],9)
        populate(b,1)
        self.assertIsNot(b.window_mask(2),mask)
        self.assertEqual(b._replay_cache_stats['mask_builds'],2)

    def test_uniform_draws_cover_only_eligible_rows(self):
        torch.manual_seed(123)
        b=ReplayBuffer(2,8,71,29,33,'cpu');bind_replay_cache(b)
        mask=torch.zeros(2,8,dtype=torch.bool);mask[0,1]=mask[0,5]=mask[1,7]=True
        e,t=b._sample_from_mask(mask,12000)
        self.assertTrue(mask[e,t].all())
        counts=torch.bincount(e*8+t,minlength=16)[mask.flatten()]
        self.assertTrue(((counts-4000).abs()<200).all())
        with self.assertRaisesRegex(RuntimeError,'no sampleable'):
            b._sample_from_mask(torch.zeros_like(mask),1)


class PrecisionTests(unittest.TestCase):
    def test_frozen_weight_cache_invalidates_after_mutation_and_training(self):
        from g1_supermimic.core.world_model.dynamics import MLP
        torch.manual_seed(9)
        model=MLP(4,(8,),2).requires_grad_(False)
        reference=deepcopy(model)
        autocast_mlp(model)
        x=torch.randn(3,4,requires_grad=True)
        for change in (0.,.2):
            with torch.no_grad():
                for a,b in zip(model.parameters(),reference.parameters()):a.add_(change);b.add_(change)
            with torch.autocast('cpu',dtype=torch.bfloat16):expected=reference(x).float()
            actual=model(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.dtype==torch.float32 for p in model.parameters()))
        model.requires_grad_(True)
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        for _ in range(2):
            opt.zero_grad();model(x).square().sum().backward();opt.step()
        model.requires_grad_(False)
        with torch.autocast('cpu',dtype=torch.bfloat16):expected=model.net(x).float()
        torch.testing.assert_close(model(x),expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA operator validation requires a GPU')
    def test_cuda_nonfinite_optimizer_block_and_bf16_backward(self):
        with TemporaryDirectory() as folder:
            tr=tiny_trainer(folder)
            for network in (tr.policy,tr.wm,tr.critic):network.cuda()
            tr.cfg=NS(performance=PerformanceCfg(fused_finite_checks=True))
            before=deepcopy(tr.policy.state_dict())
            with guard_training(tr):
                for p in tr.policy.parameters():p.grad=torch.full_like(p,float('nan'))
                with self.assertRaisesRegex(FloatingPointError,'policy gradients'):tr.policy_trainer.opt.step()
            for k,v in before.items():torch.testing.assert_close(v,tr.policy.state_dict()[k])
        for bad in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(FloatingPointError):require_finite(torch.tensor([1.,bad],device='cuda'),fused=True)
        require_finite([torch.tensor(3e38,device='cuda'),torch.tensor(complex(3e38,3e38),device='cuda')],fused=True)
        model=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,4)).cuda()
        original=deepcopy(model);autocast_mlp(model)
        x=torch.randn(8,8,device='cuda',requires_grad=True);y=x.detach().clone().requires_grad_()
        a,b=original(x),model(y)
        self.assertEqual(b.dtype,torch.float32)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(a,b,atol=.005,rtol=.05)
        torch.testing.assert_close(x.grad,y.grad,atol=.005,rtol=.05)

    def test_fused_finite_checks_match_real_complex_scalar_and_empty_cases(self):
        require_finite([torch.empty(0),torch.tensor(3e38),torch.ones(4,dtype=torch.float64),
                        torch.tensor(complex(3e38,3e38))],fused=True)
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaisesRegex(FloatingPointError,'state.bad'):
                require_finite({'good':torch.ones(2),'bad':torch.tensor([value])},fused=True)
        with TemporaryDirectory() as folder:
            tr=tiny_trainer(folder);tr.cfg=NS(performance=PerformanceCfg(fused_finite_checks=True))
            before=deepcopy(tr.policy.state_dict())
            with guard_training(tr):
                for p in tr.policy.parameters():p.grad=torch.full_like(p,float('inf'))
                with self.assertRaisesRegex(FloatingPointError,'policy gradients'):tr.policy_trainer.opt.step()
            for k,v in before.items():torch.testing.assert_close(v,tr.policy.state_dict()[k])
            baseline=tiny_trainer(folder);baseline.policy.load_state_dict(before)
            baseline.policy_trainer.update()
            with guard_training(tr):tr.policy_trainer.update()
            for k,v in baseline.policy.state_dict().items():
                torch.testing.assert_close(v,tr.policy.state_dict()[k],rtol=0,atol=0)

    def test_recurrent_world_model_keeps_fp32_states_and_policy_bptt(self):
        from test_variants import tiny_config
        from g1_supermimic.variants import make_policy,NormalizedWorldModel
        torch.manual_seed(31)
        cfg=tiny_config()
        world=NormalizedWorldModel(cfg.world_model,NS(),'cpu')
        with torch.no_grad():world.dynamics.mlp.net[-1].weight.normal_(std=.01)
        policy=make_policy(cfg)
        world.requires_grad_(False)
        mixed_world,mixed_policy=deepcopy(world),deepcopy(policy)
        autocast_network(mixed_world);autocast_network(mixed_policy)
        state=torch.zeros(3,71);state[:,3]=1
        goal=torch.randn(3,cfg.policy.goal_dim)*.1
        observation=torch.randn(3,930)*.1
        def rollout(w,p):
            s=state.clone()
            for _ in range(32):
                # Retain the recurrent state-to-action path through all 32 steps.
                history=observation+s.mean(-1,keepdim=True)
                a=p(history,goal)
                s=w.step(s,a,torch.zeros(3,33))
                self.assertEqual(s.dtype,torch.float32)
                self.assertTrue(torch.isfinite(s).all())
            loss=s.square().mean()+a.square().mean();loss.backward()
            return s
        original=rollout(world,policy)
        mixed=autocast_region(lambda:rollout(mixed_world,mixed_policy),'cpu')()
        torch.testing.assert_close(original,mixed,atol=.005,rtol=.05)
        gradients=[p.grad for p in mixed_policy.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum() for g in gradients),0)

    def test_bf16_mlp_keeps_fp32_boundary_and_finite_close_input_gradients(self):
        torch.manual_seed(23)
        original=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,4))
        mixed=deepcopy(original);autocast_mlp(mixed)
        x=torch.randn(5,8,requires_grad=True);y=x.detach().clone().requires_grad_()
        a,b=original(x),mixed(y)
        self.assertEqual(b.dtype,torch.float32)
        torch.testing.assert_close(a,b,atol=.004,rtol=.04)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=.003,rtol=.05)
        self.assertTrue(all(p.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in mixed.parameters()))

    def test_loaded_frozen_normalizer_no_longer_reads_device_counter(self):
        n=RunningNorm(2,max_count=3);n.update(torch.randn(4,2))
        restored=RunningNorm(2,max_count=3);restored.load_state_dict(n.state_dict())
        before=deepcopy(restored.state_dict())
        with patch.object(torch.Tensor,'item',side_effect=AssertionError('unexpected device sync')):
            restored.update(torch.randn(4,2))
        for k,v in before.items():torch.testing.assert_close(v,restored.state_dict()[k])

    def test_performance_options_reject_typos_and_nonboolean_switches(self):
        for options in (PerformanceCfg(1),PerformanceCfg(False,'fp16')):
            with self.assertRaises(ValueError):prepare_performance(NS(performance=options))
        cfg=NS(performance=PerformanceCfg(precision='bf16'),wm_trainer=NS(compile='default'))
        with self.assertRaisesRegex(ValueError,'requires eager'):prepare_performance(cfg)


if __name__=='__main__':unittest.main()
