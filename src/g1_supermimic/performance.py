"""Opt-in replay index caching and BF16 MLPs, without editing pinned sources."""
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from unittest.mock import patch

import torch


@dataclass
class PerformanceCfg:
    replay_index_cache: bool = False
    precision: str = 'fp32'
    fused_finite_checks: bool = False


def prepare_performance(cfg):
    if type(cfg.performance.replay_index_cache) is not bool:
        raise ValueError('performance.replay_index_cache must be a JSON boolean')
    if type(cfg.performance.fused_finite_checks) is not bool:
        raise ValueError('performance.fused_finite_checks must be a JSON boolean')
    if cfg.performance.precision not in ('fp32', 'bf16'):
        raise ValueError('performance.precision must be fp32 or bf16')
    if cfg.performance.precision == 'bf16' and any(
            getattr(getattr(cfg, section, None), 'compile', None) is not None
            for section in ('wm_trainer', 'policy_trainer')):
        raise ValueError('Local BF16 weight caching requires eager trainers; use --no-compile')
    return dict(reference='https://github.com/nico-bohlinger/xmimic',
                replay_index_cache=cfg.performance.replay_index_cache, precision=cfg.performance.precision,
                fused_finite_checks=cfg.performance.fused_finite_checks,
                bf16_weight_cache=cfg.performance.precision == 'bf16',
                sampling=('uniform eligible rows with replacement; RNG sequence differs from multinomial'
                          if cfg.performance.replay_index_cache else 'pinned upstream multinomial'),
                mixed_precision=('BF16 MLPs only; state integration, rotations, losses and parameters remain FP32'
                                 if cfg.performance.precision == 'bf16' else 'FP32'))


def bind_replay_cache(buffer):
    """Reuse eligible windows between writes, including all WM minibatches.

    Cache invalidation occurs before a mutation so even a partially failed write
    cannot leave stale indices. The upstream history/window boundary rules stay
    intact; only the uniform draw changes from multinomial to integer indices.
    """
    if hasattr(buffer, '_replay_cache_stats'):
        return
    masks, eligible = {}, {}
    stats = buffer._replay_cache_stats = dict(mask_builds=0, index_builds=0, cache_hits=0)
    original_mask = buffer.window_mask

    def invalidate(original):
        @wraps(original)
        def mutate(*a, **kw):
            masks.clear(); eligible.clear()
            return original(*a, **kw)
        return mutate
    buffer.add = invalidate(buffer.add)
    buffer.end_episode = invalidate(buffer.end_episode)

    def window_mask(self, k):
        if k not in masks:
            masks[k] = original_mask(k)
            stats['mask_builds'] += 1
        return masks[k]

    def sample(self, mask, n):
        # Keep the mask alive with its indices; id reuse cannot alias a new mask.
        entry = eligible.get(id(mask))
        if entry is None:
            idx = mask.reshape(-1).nonzero(as_tuple=False).flatten()
            stats['index_builds'] += 1
            if any(mask is cached for cached in masks.values()):
                eligible[id(mask)] = (mask, idx)
        else:
            idx = entry[1]
            stats['cache_hits'] += 1
        if idx.numel() == 0:
            raise RuntimeError('replay buffer has no sampleable rows yet')
        selected = idx[torch.randint(idx.numel(), (n,), device=idx.device)]
        return selected // self.T, selected % self.T
    buffer.window_mask = MethodType(window_mask, buffer)
    buffer._sample_from_mask = MethodType(sample, buffer)


def autocast_mlp(module):
    """Expose an FP32 boundary around a BF16 neural network, preserving BPTT."""
    original = module.forward
    device = next(module.parameters()).device.type
    cached, signature = {}, None
    linears = []
    if hasattr(module, 'net'):
        for name, child in module.net.named_modules():
            if isinstance(child, torch.nn.Linear):
                for key, parameter in child.named_parameters(recurse=False):
                    linears.append((f'{name}.{key}' if name else key, parameter))
    @wraps(original)
    def forward(*args, **kwargs):
        nonlocal cached, signature
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if linears and all(not value.requires_grad for _, value in linears):
                # Autocast does not cache frozen parameters. BPTT would otherwise
                # retain a new BF16 copy of the frozen WM at every imagined step.
                current = tuple((id(value), value._version) for _, value in linears)
                if signature != current:
                    cached = {name: value.detach().to(torch.bfloat16) for name, value in linears}
                    signature = current
                result = torch.func.functional_call(module.net, cached, args, kwargs, strict=False)
            else:
                result = original(*args, **kwargs)
        return result.float()
    module.forward = forward


def autocast_region(function, device):
    """Keep casts across a forward recurrence, with physics autocast disabled.

    Inner MLP contexts enable BF16. The disabled outer context keeps their cast
    cache alive until this function ends, then clears it before optimizer steps.
    A region must never contain optimizer updates or a no-grad/grad phase switch.
    """
    @wraps(function)
    def call(*args, **kwargs):
        with torch.autocast(device_type=device, enabled=False):
            return function(*args, **kwargs)
    return call


def bind_performance(trainer):
    options = trainer.cfg.performance
    if options.replay_index_cache:
        bind_replay_cache(trainer.buffer)
    if options.precision == 'bf16':
        # EMA is constructed before these runtime wrappers: its module stays FP32.
        networks = (trainer.policy, trainer.critic, trainer.wm)
        for network in networks:
            autocast_network(network)
        device = next(trainer.policy.parameters()).device.type
        trainer.wm_trainer.loss_on = autocast_region(trainer.wm_trainer.loss_on, device)
        from g1_supermimic.core.imagination import policy_trainer
        original_update, original_imagine = trainer.policy_trainer.update, policy_trainer.imagine
        cached_imagine = autocast_region(original_imagine, device)
        @wraps(original_update)
        def update(*args, **kwargs):
            # Critic regression has optimizer steps and must be outside the region.
            with patch.object(policy_trainer, 'imagine', cached_imagine):
                return original_update(*args, **kwargs)
        trainer.policy_trainer.update = update


def autocast_network(network):
    # MLP is the pinned module; normalizers/FK/integration are outside it.
    mlps = [module for module in network.modules() if module.__class__.__name__ == 'MLP']
    for module in mlps:
        autocast_mlp(module)
