"""Optional reference-policy action constraint around the pinned BPTT update."""
from dataclasses import asdict, dataclass
from functools import wraps
import math
import os
from pathlib import Path
from unittest.mock import patch


@dataclass
class PolicyStabilityCfg:
    reference_weight: float = 0.0
    reference_checkpoint: str | None = None
    reference_sha256: str | None = None


def prepare_stabilization(cfg, resumed, files):
    """Validate the anchor before simulation; embedded anchors survive relocation."""
    options = cfg.stability
    weight = options.reference_weight
    if (isinstance(weight, bool) or not isinstance(weight, (float, int))
            or not math.isfinite(weight) or weight < 0):
        raise ValueError('stability.reference_weight must be finite and nonnegative')
    if weight == 0:
        return None
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or cfg.policy_trainer.compile is not None:
        raise ValueError('Reference-policy stabilization currently requires single-GPU eager training')
    from .checkpoints import load_checkpoint, check_data_contract
    from .configuration import digest, merge_config
    from .variants import TrainConfig, prepare_variant
    from .safety import require_finite

    source = options.reference_checkpoint or cfg.resume
    embedded = resumed.get('policy_reference') if resumed else None
    if source is None and embedded:
        source = embedded['source_checkpoint']
    if source is None:
        raise ValueError('Reference-policy stabilization requires a reference checkpoint or --resume')
    source = str(Path(source).expanduser().resolve())
    signature = [dict(name=Path(f).name, sha256=digest(f)) for f in files]
    if embedded and source == embedded['source_checkpoint']:
        payload = embedded
        if payload['data_signature'] != signature:
            raise ValueError('Embedded reference policy motion hashes/order differ from training')
    else:
        path = Path(source)
        reference = resumed if cfg.resume is not None and path == Path(cfg.resume).resolve() else load_checkpoint(path)
        check_data_contract(path, files)
        # A raw upstream reference may not have a run manifest; verify its configured library too.
        reference_cfg = merge_config(TrainConfig(), reference['config'])
        from .configuration import data_files
        reference_files = data_files(reference_cfg)
        if [dict(name=Path(f).name, sha256=digest(f)) for f in reference_files] != signature:
            raise ValueError('Reference policy motion hashes/order differ from training')
        from .ema import policy_state
        reference_state, _ = policy_state(reference)
        payload = dict(config=reference['config'], policy=reference_state,
                       source_checkpoint=source, sha256=digest(path),
                       iteration=reference['iteration'], data_signature=signature)
    require_finite(payload, 'frozen reference policy')
    if options.reference_sha256 is not None and options.reference_sha256 != payload['sha256']:
        raise ValueError('Reference-policy SHA-256 differs from the pinned anchor')
    reference_cfg = merge_config(TrainConfig(), payload['config'])
    prepare_variant(reference_cfg)
    if (asdict(reference_cfg.policy) != asdict(cfg.policy)
            or asdict(reference_cfg.optimizations) != asdict(cfg.optimizations)
            or reference_cfg.env.future_frames != cfg.env.future_frames
            or reference_cfg.env.anchor_obs != cfg.env.anchor_obs):
        raise ValueError('Reference policy observation/network architecture differs from training')
    options.reference_checkpoint, options.reference_sha256 = source, payload['sha256']
    # Runtime-only: dataclass serialization keeps the configuration small. save() embeds this.
    cfg._policy_reference = payload
    return dict(method='reference_policy_action_mse', reference_weight=weight,
                source_checkpoint=source, source_sha256=payload['sha256'],
                reference_iteration=payload['iteration'], checkpoint_embeds_reference=True,
                samples='first real-replay state of each imagined rollout; identical noisy inputs',
                actor_loss='original -J/H + reference_weight * mean((action-reference_action)^2)',
                reference_statistics='eval mode; frozen weights and normalization buffers')


def constrained_update(update, policy, reference, weight, objective_module):
    """Reuse the original update and add MSE before its existing gradient clipping."""
    import torch
    original_objective = objective_module.policy_objective

    @wraps(update)
    def wrapped(*args, **kwargs):
        cache = {}

        def first_real_action(module, inputs, action):
            if 'mse' in cache:
                return
            with torch.no_grad():
                target = reference(*(value.detach() if torch.is_tensor(value) else value for value in inputs))
            cache['mse'] = (action - target).square().mean()

        def objective(rollout, terminal_values, gamma):
            if 'mse' not in cache:
                raise RuntimeError('Reference-policy constraint did not observe the initial real-replay action')
            value = original_objective(rollout, terminal_values, gamma)
            cache['original_objective'] = value.detach()
            # The upstream loss divides J by H. Keep the MSE coefficient independent of H.
            return value - rollout.rewards.shape[0] * weight * cache['mse']

        handle = policy.register_forward_hook(first_real_action)
        try:
            with patch.object(objective_module, 'policy_objective', objective):
                metrics = update(*args, **kwargs)
        finally:
            handle.remove()
        metrics.update(reference_action_rms=float(cache['mse'].detach().sqrt()),
                       reference_penalty_loss=float(weight * cache['mse'].detach()),
                       imag_objective_unregularized=float(cache['original_objective']))
        return metrics
    return wrapped


def embed_reference_save(save, payload):
    """Add the frozen anchor to the original save; the outer guard stays atomic."""
    import torch

    @wraps(save)
    def wrapped(*args, **kwargs):
        original_torch_save = torch.save
        def with_reference(checkpoint, *a, **kw):
            if not isinstance(checkpoint, dict) or not {'policy', 'config', 'iteration'} <= checkpoint.keys():
                raise ValueError('Unexpected checkpoint while embedding the reference policy')
            return original_torch_save(dict(checkpoint, policy_reference=payload), *a, **kw)
        with patch.object(torch, 'save', with_reference):
            return save(*args, **kwargs)
    return wrapped


def bind_stabilization(trainer):
    """Inactive by default; no inference or network-state contract changes."""
    cfg = trainer.cfg
    if cfg.stability.reference_weight == 0:
        return
    from .variants import TrainConfig, prepare_variant, make_policy
    from .configuration import merge_config
    from g1_supermimic.core.imagination import policy_trainer
    payload = cfg._policy_reference
    reference_cfg = merge_config(TrainConfig(), payload['config'])
    prepare_variant(reference_cfg)
    # Constructing the frozen network must not perturb replay sampling / training RNG.
    import torch
    with torch.random.fork_rng(devices=[]):
        reference = make_policy(reference_cfg)
    reference.load_state_dict(payload['policy'], strict=True)
    reference = reference.to(cfg.device).eval().requires_grad_(False)
    trainer.reference_policy = reference
    learner = trainer.policy_trainer
    learner.update = constrained_update(learner.update, trainer.policy, reference,
                                        cfg.stability.reference_weight, policy_trainer)
    trainer.save = embed_reference_save(trainer.save, payload)
