"""Stop at a collection boundary while retaining the configured LR horizon."""
from contextlib import contextmanager


class ShortRunComplete(Exception):
    pass


@contextmanager
def stop_after_steps(trainer, count):
    if count is None:
        yield
        return
    endpoint = trainer.iteration + count
    original = trainer.act
    def act(observation):
        if trainer.iteration >= endpoint:
            raise ShortRunComplete()
        return original(observation)
    # Stop before act(): variant actors update real-data RunningNorm here.
    trainer.act = act
    try:
        yield
    finally:
        trainer.act = original
