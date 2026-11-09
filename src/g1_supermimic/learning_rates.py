"""Keep actor and critic schedules independent without editing pinned upstream."""
from functools import wraps
import math


def base_learning_rates(cfg):
    rates = dict(world_model=cfg.wm_trainer.lr, actor=cfg.policy_trainer.lr,
                 critic=cfg.policy_trainer.critic_lr)
    for name, value in rates.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name == 'world_model' and value == 0)):
            raise ValueError(f'Invalid {name} learning rate: {value!r}')
    return rates


def bind_learning_rates(trainer):
    """Use the shared cosine factor and each optimizer's configured base LR.

    The upstream loop sets the world-model LR before calling the actor setter.
    Read that scheduled value so an intentionally frozen actor (LR=0) does not
    also freeze a nonzero critic. Optimizer moments and checkpoint keys stay
    unchanged. Equal actor/critic base rates retain the original update values.
    """
    rates = base_learning_rates(trainer.cfg)
    policy_trainer = trainer.policy_trainer
    original = policy_trainer.set_lr

    @wraps(original)
    def set_lr(actor_lr):
        original(actor_lr)
        world_lr = trainer.wm_trainer.opt.param_groups[0]['lr']
        critic_lr = world_lr * rates['critic'] / rates['world_model']
        for group in policy_trainer.critic_opt.param_groups:
            group['lr'] = critic_lr

    policy_trainer.set_lr = set_lr
