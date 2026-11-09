"""One training/benchmark job per project output directory."""
from contextlib import contextmanager
from ..paths import ROOT

@contextmanager
def training_lock():
    import fcntl
    path=ROOT/'.cache/training.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another training/benchmark process is using this project') from None
        yield
