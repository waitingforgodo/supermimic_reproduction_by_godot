"""资源监测与训练日志汇总，不参与 PPO 数学更新。

nvidia-smi 统计整块显卡，/proc/meminfo 统计整个 WSL 实例。它们不是本进程
的独占占用，约一秒的采样也可能漏掉瞬时峰值，结果应按资源余量理解。
"""
import json
import math
import statistics
import subprocess
import threading
from pathlib import Path


def available_ram_gib():
    """读取 Linux MemAvailable，优先反映可供新任务使用的内存而非裸 free。"""
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return None


def gpu_free_mib():
    """读取第 0 块 GPU 的剩余显存；失败返回 None，不把缺数据当无限余量。"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', '0'],
                                text=True, capture_output=True, timeout=4, check=True)
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class ResourceSampler:
    """后台每秒采样一次，主线程训练继续运行；退出后汇总采样最小余量。"""
    def __init__(self):
        """只建立线程对象，调用 start 才开始读资源。"""
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        """使用 Event.wait 以便退出可及时唤醒；单次 nvidia-smi 有超时限制。"""
        while not self.stop_event.is_set():
            self.samples.append((gpu_free_mib(), available_ram_gib()))
            self.stop_event.wait(1)

    def start(self):
        """在环境初始化前开始采样，包含初始化/编译/训练阶段的资源变化。"""
        self.thread.start()

    def finish(self):
        """停止采样并返回最低余量；无有效数据时保留 None。"""
        self.stop_event.set()
        self.thread.join(timeout=6)
        gpu = [g for g, _ in self.samples if g is not None]
        ram = [r for _, r in self.samples if r is not None]
        return {'min_free_vram_mib': min(gpu) if gpu else None,
                'min_available_ram_gib': min(ram) if ram else None,
                'resource_samples': len(self.samples), 'resource_scope': 'whole GPU and WSL, not just this process'}


def timing_statistics(fps, collection_times, learning_times, warmup=5):
    """按 PPO 更新编号对齐三种计时，丢弃预热后检查负数/缺项/非有限值。

    一个负耗时就能说明测量不可靠；不能因为整组中位数为正就静默通过。
    续训编号不从 0 开始，故跳过前五条记录，而不是跳过编号小于五的记录。
    """
    checked, valid_fps, invalid_steps, missing_steps = 0, [], [], []
    for step, value in sorted(fps.items())[warmup:]:
        if step not in collection_times or step not in learning_times:
            missing_steps.append(step)
            continue
        checked += 1
        collection, learning = collection_times[step], learning_times[step]
        if (not all(math.isfinite(x) for x in (value, collection, learning))
                or value <= 0 or collection < 0 or learning < 0 or collection + learning <= 0):
            invalid_steps.append(step)
            continue
        valid_fps.append(value)
    return dict(median_fps=statistics.median(valid_fps) if valid_fps else 0,
                timing_valid=bool(valid_fps) and not invalid_steps and not missing_steps,
                checked_timing_updates=checked, valid_timing_updates=len(valid_fps),
                invalid_timing_steps=invalid_steps, missing_timing_steps=missing_steps,
                timing_warning='Invalid or incomplete elapsed times; rerun before choosing parallelism.'
                if invalid_steps or missing_steps or not valid_fps else None)


def summarize(run):
    """读取 event 文件与资源 JSON，不载入 checkpoint，也不执行任何训练。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = Path(run)
    accumulator = EventAccumulator(str(run), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = accumulator.Tags().get('scalars', [])
    last = {}
    for tag in tags:
        points = accumulator.Scalars(tag)
        if points:
            last[tag] = points[-1].value
    def values(tag):
        """以 step 为键对齐同一轮数据；重复写入同一编号时保留最后记录。"""
        return {p.step: p.value for p in accumulator.Scalars(tag)} if tag in tags else {}
    fps_values = values('Perf/total_fps')
    timings = timing_statistics(fps_values, values('Perf/collection_time'), values('Perf/learning_time'))
    result = {'run': str(run), 'last_metrics': last, 'measured_updates': len(fps_values), **timings,
              'note': 'Runtime success does not imply a converged gait.'}
    resource_file = run / 'resources.json'
    if resource_file.exists():
        result.update(json.loads(resource_file.read_text()))
    return result
