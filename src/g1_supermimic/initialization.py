"""Initialize a new motion task from learned networks, without resuming its state."""
from dataclasses import asdict


def prepare_initialization(cfg, checkpoint, source):
    from .configuration import digest, merge_config
    from .variants import TrainConfig, prepare_variant
    from .ema import policy_state

    original = merge_config(TrainConfig(), checkpoint['config'])
    prepare_variant(original)
    for key in ('policy', 'critic', 'world_model'):
        if asdict(getattr(original, key)) != asdict(getattr(cfg, key)):
            raise ValueError(f'--init-from requires the same {key} architecture and feature configuration')
    state, policy_source = policy_state(checkpoint)
    cfg._init_from = source
    # Retain only CPU network tensors; old optimizer/sampler state is discarded.
    cfg._initial_weights = dict(wm=checkpoint['wm'], policy=state, critic=checkpoint['critic'])
    return dict(method='network_weights_only', source_checkpoint=str(source),
                source_sha256=digest(source), source_iteration=checkpoint['iteration'],
                source_updates=checkpoint.get('num_updates'), policy_source=policy_source,
                restored=['world_model', 'policy', 'critic', 'normalization_buffers'],
                reset=['optimizers', 'replay', 'sampler', 'EMA_history', 'style_discriminator', 'iteration', 'num_updates', 'selection'],
                initial_evaluation='all target motions before the first training step; old dataset scores are not reused')


def apply_initialization(trainer):
    states = getattr(trainer.cfg, '_initial_weights', None)
    if states is None:
        return
    for name, state in states.items():
        getattr(trainer, name).load_state_dict(state, strict=True)
    del trainer.cfg._initial_weights
    if trainer.is_main:
        print(f'[INFO] Initialized networks from {trainer.cfg._init_from}; '
              'new task starts at step 0 with fresh optimizers and sampler', flush=True)
