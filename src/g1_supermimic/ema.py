"""Optional online policy averaging; retain the online optimizer state for resume."""
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import math
from unittest.mock import patch

import torch


@dataclass
class PolicyEMACfg:
    decay: float = 0.0  # zero keeps legacy training/inference unchanged


def prepare_ema(cfg, resumed=None):
    decay = cfg.ema.decay
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError('ema.decay must be finite and in [0, 1)')
    previous = resumed and resumed['config'].get('ema', {}).get('decay', 0)
    if previous and not decay:
        raise ValueError('Cannot disable EMA when resuming an EMA-selected checkpoint; use its original online parent for ablation')
    if previous and 'policy_ema' not in resumed:
        raise ValueError('EMA checkpoint is missing policy_ema')
    cfg._ema_resume = resumed.get('policy_ema') if resumed and decay else None
    cfg._ema_updates = resumed.get('ema_updates', 0) if resumed and decay else 0
    if type(cfg._ema_updates) is not int or cfg._ema_updates < 0:
        raise ValueError('Invalid ema_updates')
    if not decay:
        return None
    return dict(reference='https://github.com/nico-bohlinger/xmimic',
                decay=decay, parameters='exponential moving average after every successful policy optimizer step',
                buffers='copy current online buffers; normalization moments and integer counters are not averaged',
                evaluation_policy='ema', resume_policy='online',
                initialized_from='saved EMA' if cfg._ema_resume is not None else 'online policy')


def policy_state(checkpoint, source='auto'):
    if source not in ('auto', 'online', 'ema'):
        raise ValueError(f'Unknown policy source: {source}')
    if source == 'auto':
        source = 'ema' if checkpoint['config'].get('ema', {}).get('decay', 0) else 'online'
    if source == 'ema' and 'policy_ema' not in checkpoint:
        raise ValueError('Checkpoint has no EMA policy')
    return checkpoint['policy_ema' if source == 'ema' else 'policy'], source


class PolicyEMA:
    def __init__(self, policy, decay, state=None, updates=0):
        self.online = policy
        self.module = deepcopy(policy).eval().requires_grad_(False)
        self.decay, self.updates = decay, updates
        if state is not None:
            self.module.load_state_dict(state, strict=True)

    @torch.no_grad()
    def copy_buffers(self):
        target = dict(self.module.named_buffers())
        for name, value in self.online.named_buffers():
            target[name].copy_(value)

    @torch.no_grad()
    def update(self):
        # State starts at the online parent, so even the first step is smoothed.
        torch._foreach_lerp_(list(self.module.parameters()), list(self.online.parameters()), 1 - self.decay)
        self.copy_buffers()
        self.updates += 1

    def state_dict(self):
        return self.module.state_dict()


def bind_ema(trainer):
    decay = trainer.cfg.ema.decay
    if not decay:
        return
    if not hasattr(trainer.cfg, '_ema_resume') and trainer.cfg.resume is not None:
        from .checkpoints import load_checkpoint
        prepare_ema(trainer.cfg, load_checkpoint(trainer.cfg.resume))
    average = trainer.policy_ema = PolicyEMA(trainer.policy, decay,
        getattr(trainer.cfg, '_ema_resume', None), getattr(trainer.cfg, '_ema_updates', 0))
    trainer.policy_trainer.opt.register_step_post_hook(lambda *unused: average.update())
    original = trainer.save

    @wraps(original)
    def save(*args, **kwargs):
        average.copy_buffers()
        torch_save = torch.save
        def enriched(checkpoint, *a, **kw):
            return torch_save(dict(checkpoint, policy_ema=average.state_dict(), ema_updates=average.updates), *a, **kw)
        with patch.object(torch, 'save', enriched):
            return original(*args, **kwargs)
    trainer.save = save
