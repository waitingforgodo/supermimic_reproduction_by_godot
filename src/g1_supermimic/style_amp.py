"""Optional real-data AMP discriminator and differentiable BPTT style reward.

Method reference: Renforce-Dynamics/trackerlab-supermimic, commit
3b8e31ea36183c4ccc1f5c3a023bfa292d141692. This adapter retains the local
reference-conditioned Actor and physical tracking benchmark. It is not the
reference project's command-locomotion policy or its terrain implementation.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import math
import os
from unittest.mock import patch

import torch
from torch import nn

REFERENCE = dict(repository='https://github.com/Renforce-Dynamics/trackerlab-supermimic',
                 commit='3b8e31ea36183c4ccc1f5c3a023bfa292d141692',
                 document='docs/supermimic_command_style.md')
FRAME_DIM = 67


@dataclass
class StyleAMPCfg:
    weight: float = 0.0  # reward per second; multiplied by the 0.02 s control step
    history: int = 10
    hidden: tuple[int, ...] = (256, 128)
    lr: float = 1.e-4
    batch_size: int = 256
    updates_per_policy: int = 1
    gradient_penalty: float = 5.0
    weight_decay: float = 3.e-4
    logit_regularization: float = .04


def prepare_style_amp(cfg, resumed, files):
    options = cfg.style_amp
    for key in ('weight', 'lr', 'gradient_penalty', 'weight_decay', 'logit_regularization'):
        value = getattr(options, key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'style_amp.{key} must be finite and nonnegative')
    for key in ('history', 'batch_size', 'updates_per_policy'):
        if type(getattr(options, key)) is not int or getattr(options, key) < 1:
            raise ValueError(f'style_amp.{key} must be a positive integer')
    if not 2 <= options.history <= cfg.policy.history:
        raise ValueError('style_amp.history must be between 2 and the Actor history length')
    if not options.hidden or any(type(w) is not int or w <= 0 for w in options.hidden):
        raise ValueError('style_amp.hidden must contain positive integer widths')
    if not options.weight:
        return None
    if not options.lr:
        raise ValueError('Enabled style_amp requires a positive discriminator learning rate')
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1
            or cfg.policy_trainer.compile is not None or cfg.wm_trainer.compile is not None
            or cfg.performance.precision != 'fp32'):
        raise ValueError('Local AMP style currently requires single-GPU FP32 eager training')
    from .configuration import digest
    signature = [[p.name, digest(p)] for p in files]
    previous = bool(resumed and resumed['config'].get('style_amp', {}).get('weight', 0))
    saved = resumed.get('style_amp') if resumed and previous else None
    if previous and saved is None:
        raise ValueError('AMP-style checkpoint is missing discriminator/optimizer state')
    if saved is not None:
        if (saved.get('version') != 1 or saved.get('expert_signature') != signature
                or saved.get('history') != options.history or saved.get('hidden') != list(options.hidden)):
            raise ValueError('AMP-style checkpoint expert data or discriminator layout differs')
    cfg._style_signature, cfg._style_resume = signature, saved
    return dict(**REFERENCE, method='least_squares_AMP_auxiliary_for_reference_tracking',
                features='gravity(3), joint offsets(29), joint velocities*0.05(29), body linear/angular velocities(6)',
                frame_dim=FRAME_DIM, history=options.history, expert_signature=signature,
                discriminator_samples='detached complete real-replay windows; no terminals, resets or clip boundaries',
                normalization='fixed expert population mean/std, minimum std=0.1; read-only while scoring',
                actor_reward='original tracking reward + weight * control_dt * clamp(1 - (D-1)^2/4, min=0)',
                weight_per_second=options.weight, physical_reward_and_best_selection='unchanged tracking only',
                actor_input_and_architecture='unchanged; reference motion still required',
                actor_start_sampling='complete real history required; sampling distribution differs from legacy starts',
                initialized_from='saved discriminator and optimizer' if saved is not None else 'fresh discriminator',
                replay='reuse physical replay buffer; refill after resume')


def frame_features(state, default_joint_pos):
    """Invariant 67-d features of local 71-d states; angular velocity is body-frame."""
    from g1_supermimic.core.kinematics.rotations import projected_gravity, quat_apply_inverse
    q = state[..., 3:7]
    return torch.cat((projected_gravity(q), state[..., 7:36]-default_joint_pos,
                      state[..., 42:71]*.05, quat_apply_inverse(q, state[..., 36:39]),
                      state[..., 39:42]), -1)


def complete_window_mask(buffer, history):
    """Episode-interior complete state windows; the final row must not be terminal."""
    return buffer.window_mask(history-1) & ~buffer._rolled(buffer.terminal, history-1)


def real_windows(buffer, history, count):
    mask = complete_window_mask(buffer, history)
    e, t = buffer._sample_from_mask(mask, count)
    indices = (t[:, None]+torch.arange(history, device=t.device)) % buffer.T
    return buffer.s[e[:, None], indices].detach()


class StyleAMP(nn.Module):
    def __init__(self, options, library, signature, saved=None):
        super().__init__()
        self.options, self.signature = options, signature
        self.updates = 0
        device = library.default_joint_pos.device
        self.register_buffer('default_joint_pos', library.default_joint_pos.detach().clone().float())
        from g1_supermimic.core.kinematics.rotations import quat_apply_inverse
        with torch.no_grad():
            ref = library.ref71(torch.arange(library.num_frames, device=device)).float()
            # NPZ/library stores WORLD angular velocities; simulator state stores BODY angular velocities.
            states = torch.cat((ref[..., :39], quat_apply_inverse(ref[..., 3:7], ref[..., 39:42]),
                                ref[..., 42:]), -1)
            experts = frame_features(states, self.default_joint_pos).clamp(-1.e4, 1.e4)
            start = torch.arange(max(0, library.num_frames-options.history+1), device=device)
            start = start[library.frame_clip[start] == library.frame_clip[start+options.history-1]]
            if not start.numel():
                raise ValueError('AMP experts have no complete within-clip history windows')
            self.register_buffer('mean', experts.mean(0))
            self.register_buffer('std', experts.var(0, unbiased=False).sqrt().clamp_min(.1))
            self.register_buffer('_experts', experts, persistent=False)
            self.register_buffer('_expert_starts', start, persistent=False)
        layers, width = [], FRAME_DIM*options.history
        for hidden in options.hidden:
            layers.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.discriminator = nn.Sequential(*layers).to(device)
        with torch.no_grad():
            layers[-1].weight.mul_(.1)
            layers[-1].bias.zero_()
        self.opt = torch.optim.AdamW(self.discriminator.parameters(), lr=options.lr,
                                     weight_decay=options.weight_decay)
        if saved is not None:
            if (saved.get('version') != 1 or saved['expert_signature'] != signature or saved['history'] != options.history
                    or saved['hidden'] != list(options.hidden)):
                raise ValueError('AMP-style state does not match expert data or layout')
            if not torch.equal(saved['model']['default_joint_pos'].to(device), self.default_joint_pos):
                raise ValueError('AMP-style robot default pose differs')
            self.load_state_dict(saved['model'], strict=True)
            if not bool((self.std >= .1).all()):
                raise ValueError('Invalid AMP-style normalization standard deviation')
            self.opt.load_state_dict(saved['optimizer'])
            for group in self.opt.param_groups:
                group['lr'], group['weight_decay'] = options.lr, options.weight_decay
            self.updates = saved['updates']
            if type(self.updates) is not int or self.updates < 0:
                raise ValueError('Invalid AMP-style update counter')

    def normalized(self, frames):
        value = ((frames.float().clamp(-1.e4, 1.e4)-self.mean)/self.std).clamp(-10, 10)
        return value.flatten(-2)

    @torch.no_grad()
    def expert_windows(self, count):
        indices = self._expert_starts[torch.randint(len(self._expert_starts), (count,), device=self.mean.device)]
        offsets = torch.arange(self.options.history, device=self.mean.device)
        return self._experts[indices[:, None]+offsets]

    def style_reward(self, frames):
        with torch.autocast(device_type=frames.device.type, enabled=False):
            score = self.discriminator(self.normalized(frames)).squeeze(-1)
        return (1-.25*(score-1).square()).clamp_min(0)

    @contextmanager
    def frozen(self):
        parameters = list(self.discriminator.parameters())
        flags = [p.requires_grad for p in parameters]
        try:
            self.discriminator.requires_grad_(False)
            yield
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)

    def update(self, buffer):
        from .safety import require_finite
        options = self.options
        losses, penalties, real_scores, expert_scores = [], [], [], []
        for _ in range(options.updates_per_policy):
            with torch.no_grad():
                policy = frame_features(real_windows(buffer, options.history, options.batch_size), self.default_joint_pos)
                expert = self.expert_windows(options.batch_size)
                inputs = torch.cat((self.normalized(policy), self.normalized(expert)), 0).detach()
            require_finite(inputs, 'AMP real/expert windows')
            inputs.requires_grad_(True)
            scores = self.discriminator(inputs).squeeze(-1)
            negative, positive = scores.split(options.batch_size)
            gradient = torch.autograd.grad(scores.sum(), inputs, create_graph=True)[0]
            penalty = gradient.square().sum(-1).mean()
            loss = .5*((negative+1).square().mean()+(positive-1).square().mean())
            loss = loss + options.gradient_penalty*penalty
            loss = loss + options.logit_regularization*self.discriminator[-1].weight.square().sum()
            require_finite(loss, 'AMP discriminator loss')
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10., error_if_nonfinite=True)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.updates += 1
            losses.append(loss.detach()); penalties.append(penalty.detach())
            real_scores.append(negative.detach().mean()); expert_scores.append(positive.detach().mean())
        return dict(amp_loss=float(torch.stack(losses).mean()),
                    amp_gradient_penalty=float(torch.stack(penalties).mean()),
                    amp_policy_score=float(torch.stack(real_scores).mean()),
                    amp_expert_score=float(torch.stack(expert_scores).mean()),
                    amp_updates=self.updates)

    def add_rollout_reward(self, rollout, start, control_dt):
        history = self.options.history
        before = frame_features(start.hist_states[-history:], self.default_joint_pos)
        following = frame_features(rollout.states[1:], self.default_joint_pos)
        sequence = torch.cat((before[1:], following), 0)
        windows = sequence.unfold(0, history, 1).permute(0, 1, 3, 2)
        style = self.style_reward(windows)
        rollout.rewards = rollout.rewards + self.options.weight*control_dt*style
        rollout.reward_terms = dict(rollout.reward_terms, amp_style_mean=float(style.detach().mean()))
        return rollout

    def checkpoint(self):
        return dict(version=1, model=self.state_dict(), optimizer=self.opt.state_dict(), updates=self.updates,
                    expert_signature=self.signature, history=self.options.history, hidden=list(self.options.hidden),
                    replay_restored=False, replay_refill_required=True)


def bind_style_amp(trainer):
    if not trainer.cfg.style_amp.weight:
        return
    # A new discriminator must not alter the real simulator/policy initialization RNG.
    devices = [trainer.device] if torch.device(trainer.device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(trainer.cfg.seed+1907)
        style = StyleAMP(trainer.cfg.style_amp, trainer.lib, trainer.cfg._style_signature,
                         getattr(trainer.cfg, '_style_resume', None))
    trainer.style_amp = style
    original_update = trainer.policy_trainer.update
    from g1_supermimic.core.imagination import policy_trainer

    @wraps(original_update)
    def update(buffer, *args, **kwargs):
        metrics = style.update(buffer)
        original_imagine = policy_trainer.imagine  # compose with runtime autocast/other wrappers
        original_starts = buffer.sample_starts

        def starts(count, history):
            sample = buffer._sample_from_mask
            full = torch.roll(complete_window_mask(buffer, style.options.history), style.options.history-1, 1)
            def complete(mask, n):
                return sample(mask & full, n)
            with patch.object(buffer, '_sample_from_mask', complete):
                return original_starts(count, history)

        def imagine(wm, policy, lib, start, *a, **kw):
            rollout = original_imagine(wm, policy, lib, start, *a, **kw)
            return style.add_rollout_reward(rollout, start, wm.spec.dt)

        with style.frozen(), patch.object(buffer, 'sample_starts', starts), patch.object(policy_trainer, 'imagine', imagine):
            result = original_update(buffer, *args, **kwargs)
        return dict(result, **metrics)
    trainer.policy_trainer.update = update
    original_save = trainer.save

    @wraps(original_save)
    def save(*args, **kwargs):
        torch_save = torch.save
        def enriched(checkpoint, *a, **kw):
            return torch_save(dict(checkpoint, style_amp=style.checkpoint()), *a, **kw)
        with patch.object(torch, 'save', enriched):
            return original_save(*args, **kwargs)
    trainer.save = save
