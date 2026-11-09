"""上游源码导入与原运行环境初始化；不执行依赖安装。"""
import os
from pathlib import Path
import sys
from .paths import ROOT


def _configure_torch29_backends(allow_tf32=True, deterministic=False):
    """2.9 的 Inductor 仍读取旧 TF32 API；所有写入统一到同一 API。"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def configure_backend_compatibility():
    """仅修正当前 SuperMimic 进程的 mjlab 后端设置，保持 TF32 含义。"""
    import torch
    if torch.__version__.startswith('2.9.'):
        import mjlab.utils.torch as mjlab_torch
        mjlab_torch.configure_torch_backends = _configure_torch29_backends
        return 'torch-2.9: process-local mjlab TF32 configuration uses allow_tf32 for Inductor compatibility'
    return None


def activate():
    """The core is installed as part of this package; no external source checkout."""
    import g1_supermimic.core  # noqa: F401


def initialize():
    """使用原运行环境，把运行缓存放回当前工作空间。没有包管理操作。"""
    activate()
    import torch
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / '.cache/warp')
    torch.set_num_threads(int(os.environ.get('G1_CPU_THREADS', '2')))
    configure_backend_compatibility()
