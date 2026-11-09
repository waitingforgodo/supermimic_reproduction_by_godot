"""Optional horizon-normalized reset returns, following NJU's sampler correction."""
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from unittest.mock import patch
import math

VISITS_SATURATED = 2**30


@dataclass
class SamplingCfg:
    horizon_returns: bool = False
    count_warmstart: bool = False
    hard_clip_cap: float = 0.


def prepare_sampling(cfg, resumed=None, *, num_clips=None):
    if any(type(getattr(cfg.sampling, key)) is not bool for key in ('horizon_returns', 'count_warmstart')):
        raise ValueError('sampling options must be JSON booleans')
    cap = cfg.sampling.hard_clip_cap
    if isinstance(cap, bool) or not isinstance(cap, (float, int)) or not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError('sampling.hard_clip_cap must be a finite number in [0, 1]')
    if cfg.sampling.count_warmstart and not cfg.sampling.horizon_returns:
        raise ValueError('count_warmstart requires horizon_returns to avoid warming up biased returns')
    if cap > 0 and not cfg.sampling.horizon_returns:
        raise ValueError('hard_clip_cap requires horizon_returns')
    if cap > 0 and num_clips is not None and cap * num_clips < 1 - 1e-7:
        raise ValueError('hard_clip_cap is infeasible for this number of clips')
    previous = bool(resumed and resumed['config'].get('sampling', {}).get('horizon_returns', False))
    if previous and not cfg.sampling.horizon_returns:
        raise ValueError('Cannot reuse normalized sampler statistics with horizon_returns disabled')
    cfg._sampling_reset_on_start = cfg.sampling.horizon_returns and not previous
    cfg._sampling_visits = resumed.get('bin_visits') if resumed is not None else None
    if cfg._sampling_visits is not None:
        import torch
        visits = cfg._sampling_visits
        if (not torch.is_tensor(visits) or visits.ndim != 1
                or visits.dtype not in (torch.int32, torch.int64)
                or (visits < 0).any() or (visits > VISITS_SATURATED).any()):
            raise ValueError('Invalid bin_visits in resumed sampler state')
    if not cfg.sampling.horizon_returns:
        return None
    return dict(horizon_returns=True, reset_sampler_state=cfg._sampling_reset_on_start,
                reference_repository='https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real',
                reference_commit='f8f60d7bbdba68468c050ef100af02d752ca238e',
                visit_count_warmstart=cfg.sampling.count_warmstart,
                hard_clip_cap=cap,
                visit_counter_initialization=('zero' if cfg._sampling_reset_on_start else
                                              'checkpoint' if cfg._sampling_visits is not None else 'saturated'),
                scope='Horizon-normalized return; optional count warmstart and per-clip hard cap; preserve 50/40/10 mixture')


def cap_per_clip(hard, bin_clip, num_clips, cap):
    """NJU-style water filling; preserve relative bin weights within each clip.

    Also handle zero softmax mass and the exactly-uniform cap explicitly.
    Reference MIT license: ../supermimic_nju/upstream/LICENSE.
    """
    import torch
    if cap <= 0 or cap >= 1:
        return hard
    if cap * num_clips < 1 - 1e-7:
        raise ValueError('hard_clip_cap is infeasible for this number of clips')
    p = hard.double().clamp_min(1e-30)
    p = p / p.sum()
    mass = lambda: torch.zeros(num_clips, dtype=p.dtype, device=p.device).index_add_(0, bin_clip, p)
    if cap * num_clips <= 1 + 1e-7:
        return (p / (num_clips * mass()[bin_clip])).to(hard.dtype)
    capped = torch.zeros(num_clips, dtype=torch.bool, device=p.device)
    for _ in range(num_clips):
        masses = mass()
        over = (masses > cap * (1 + 1e-7)) & ~capped
        if not over.any():
            break
        capped |= over
        scale = torch.ones_like(masses)
        scale[capped] = cap / masses[capped]
        scale[~capped] = (1 - cap * capped.sum().to(p.dtype)) / masses[~capped].sum()
        p = p * scale[bin_clip]
    return p.to(hard.dtype)


def bind_clip_cap(command, cap):
    """Preserve upstream reset modes/frame bounds, cap only the hard branch."""
    if cap <= 0:
        return
    import torch
    lib = command.library
    if cap * lib.num_clips < 1 - 1e-7:
        raise ValueError('hard_clip_cap is infeasible for this number of clips')
    original = command._sample_frames

    def sample(self, n):
        cfg = self.cfg
        if cfg.sampling_mode != 'return':
            return original(n)
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        hard = cap_per_clip(hard, lib.bin_clip, lib.num_clips, cap)
        self.bin_memory = (1-cfg.memory_alpha)*self.bin_memory + cfg.memory_alpha*hard
        p = cfg.uniform_ratio/lib.num_bins + cfg.hard_ratio*hard + cfg.memory_ratio*self.bin_memory/self.bin_memory.sum()
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (lib.bin_end[b]-lib.bin_start[b]).float()).long()
        f = torch.minimum(f, lib.last_frame_of(f)-2).clamp_min(lib.clip_start[lib.frame_clip[f]])
        ent = -(p * (p+1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        self.metrics['sampling_entropy'][:] = ent
        return f
    command._sample_frames = MethodType(sample, command)


def embed_sampler_save(save, command):
    """Persist counters through the original save and the outer atomic guard."""
    import torch
    @wraps(save)
    def wrapped(*args, **kwargs):
        # Respect other local checkpoint wrappers active around this save.
        original = torch.save
        def with_visits(checkpoint, *a, **kw):
            if not isinstance(checkpoint, dict) or not {'policy', 'config', 'iteration'} <= checkpoint.keys():
                raise ValueError('Unexpected checkpoint while embedding sampler counters')
            return original(dict(checkpoint, bin_visits=command.bin_visits), *a, **kw)
        with patch.object(torch, 'save', with_visits):
            return save(*args, **kwargs)
    return wrapped


def normalized_returns(rtg, lens, frames, terminated, clip_end, last_frame,
                       *, gamma, max_step_reward, g_max, bootstrapped=False):
    """A fall loses future rewards; it must not normalize to perfect tracking."""
    import torch
    t = torch.arange(rtg.shape[1], device=rtg.device)[None, :]
    steps_left = (lens[:, None] - t).clamp_min(1)
    frames_left = last_frame - frames + 2
    fallen = terminated & ~clip_end
    horizon = torch.where(fallen[:, None], frames_left, steps_left).double()
    achievable = max_step_reward * (-torch.expm1(horizon * math.log(gamma))) / (1 - gamma)
    scale = g_max / achievable
    if bootstrapped:
        truncated = ~terminated & ~clip_end
        scale = torch.where(truncated[:, None], torch.ones_like(scale), scale)
    return rtg * scale


def bind_sampling(trainer):
    """Bind only the live training command; the pinned source stays unchanged."""
    if not trainer.cfg.sampling.horizon_returns:
        return
    import torch
    command = trainer.command
    command.ep_frames = torch.zeros_like(command.ep_bins)
    command.bin_visits = torch.zeros(command.library.num_bins, dtype=torch.int32, device=command.device)
    if getattr(trainer.cfg, '_sampling_reset_on_start', True):
        command.bin_return.fill_(command.cfg.g_max)
        command.bin_memory.fill_(1. / command.library.num_bins)
    else:
        visits = getattr(trainer.cfg, '_sampling_visits', None)
        if visits is None:
            # Old normalized checkpoints retain their mature fixed-rate EMA.
            command.bin_visits.fill_(VISITS_SATURATED)
        else:
            if visits.shape != command.bin_visits.shape:
                raise ValueError('Sampler visit-counter shape differs from motion library')
            command.bin_visits.copy_(visits)
    log = command._log_step_rewards

    def log_frames(self, env_mask=None):
        if getattr(self._env, 'reward_buf', None) is None:
            return
        ids = (torch.arange(self.num_envs, device=self.device) if env_mask is None
               else env_mask.nonzero().squeeze(-1))
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        log(env_mask)
        self.ep_frames[ids, pos] = self.frame[ids]

    def finalize(self, env_ids):
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max().item())
        mask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        reward = self.ep_rewards[env_ids, :T].double() * mask
        gp = self._gamma_pow[:T]
        rtg = torch.flip(torch.cumsum(torch.flip(reward * gp, [1]), 1), [1]) / gp
        manager = getattr(self._env, 'termination_manager', None)
        terminated = manager.terminated[env_ids] if manager is not None else torch.ones_like(lens, dtype=torch.bool)
        ended = self.clip_end[env_ids]
        truncated = ~terminated & ~ended
        if truncated.any():
            remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0)
            if self.bootstrap_fn is not None:
                value = self.bootstrap_fn(env_ids).double()
                rtg = rtg + truncated[:, None] * cfg.gamma ** remaining.double() * value[:, None]
            else:
                mask = mask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        frames = self.ep_frames[env_ids, :T]
        rtg = normalized_returns(rtg, lens, frames, terminated, ended,
                                 self.library.last_frame_of(frames), gamma=cfg.gamma,
                                 max_step_reward=trainer.max_step_reward, g_max=cfg.g_max,
                                 bootstrapped=self.bootstrap_fn is not None)
        valid = mask.reshape(-1)
        bins = self.ep_bins[env_ids, :T].reshape(-1)[valid]
        values = rtg.reshape(-1)[valid].float()
        if bins.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, bins, values)
        counts = torch.zeros_like(sums).index_add_(0, bins, torch.ones_like(values))
        touched = counts > 0
        mean, old = sums[touched]/counts[touched], self.bin_return[touched]
        visits = self.bin_visits[touched]
        updated = (1-cfg.ema_alpha)*old + cfg.ema_alpha*mean
        if trainer.cfg.sampling.count_warmstart:
            weight = 1. / (visits.float() + 1.)
            updated = torch.where(weight > cfg.ema_alpha, (1-weight)*old + weight*mean, updated)
        self.bin_return[touched] = updated
        self.bin_visits[touched] = (visits + 1).clamp_max(VISITS_SATURATED)
        self.metrics['episode_return_to_go'][env_ids] = rtg[:, 0].float()
        self.ep_len[env_ids] = 0

    command._log_step_rewards = MethodType(log_frames, command)
    command._finalize_episodes = MethodType(finalize, command)
    bind_clip_cap(command, trainer.cfg.sampling.hard_clip_cap)
    trainer.save = embed_sampler_save(trainer.save, command)
