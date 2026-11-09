"""固定版本 RSL-RL 的耗时兼容层，解决系统时钟回拨造成负耗时的问题。"""
from contextlib import contextmanager


class _ElapsedClock:
    """只把 runner 调用的 time() 换成单调时钟，其余属性仍交给原 time 模块。"""
    def __init__(self, clock):
        """保存原时钟模块，避免改写全局标准库 time.time。"""
        self.clock = clock

    def time(self):
        """用于相减求间隔，绝对数值不应被当作 Unix 时间戳显示。"""
        return self.clock.perf_counter()

    def __getattr__(self, name):
        """委托未覆盖方法，保持原模块其他能力。"""
        return getattr(self.clock, name)


@contextmanager
def monotonic_runner_clock(runner_module):
    """只改当前进程一个 runner 模块的 time 引用，结束或异常时均恢复。

    本空间每个进程只运行一个 learner。TensorBoard 引用的标准库时间没有
    被修改，所以 event 的日期时间仍是真实时间；依赖目录也没有文件改动。
    """
    original = runner_module.time
    runner_module.time = _ElapsedClock(original)
    try:
        yield
    finally:
        runner_module.time = original
