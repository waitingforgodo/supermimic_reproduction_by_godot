"""SuperMimic 参数、数据校验和来源记录；帮助只依赖标准库。"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .paths import ROOT


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def train_parser():
    p = argparse.ArgumentParser(description='Train Godot SuperMimic reference-conditioned G1 tracking')
    p.add_argument('--config', type=Path, help='Partial JSON applied after the local profile or saved checkpoint; unknown fields are rejected')
    data = p.add_mutually_exclusive_group()
    data.add_argument('--motion', type=Path, help='One 50 Hz NPZ for a new run; resume retains the data contract')
    data.add_argument('--npz-dir', type=Path, help='50 Hz NPZ directory; default 40-clip LAFAN1 dataset')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--strict-upstream', dest='strict_upstream', action='store_true', default=False,
                      help='Use pinned upstream defaults instead of the local fresh-run profile; validate full LAFAN1')
    mode.add_argument('--allow-config-changes', dest='strict_upstream', action='store_false',
                      help='Allow local/custom settings (default); deviations are always recorded')
    p.add_argument('--clips', nargs='+', help='Clip stems within --npz-dir')
    p.add_argument('--envs', type=positive, help='Parallel simulation environments; upstream default 4096')
    p.add_argument('--iterations', type=positive, help='TOTAL vectorized env steps, not optimizer updates; default 320000')
    p.add_argument('--stop-after-steps', type=positive,
                   help='Short comparison: stop after this many additional control steps, preserving the saved LR schedule')
    p.add_argument('--name', help='New run directory name; default timestamp')
    p.add_argument('--run-root', type=Path, help='Separate experiment directory; defaults to normal runs (and latest)')
    p.add_argument('--seed', type=int)
    p.add_argument('--device')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--resume', type=Path, help='Restore model, optimizers and counters on the same motion set')
    start.add_argument('--init-from', type=Path,
                       help='Initialize networks/normalization from a checkpoint on explicit --motion or --npz-dir; reset optimizers, sampler and step counters')
    p.add_argument('--replay-capacity', type=positive)
    p.add_argument('--world-batch-size', type=positive)
    p.add_argument('--world-batches-per-update', type=positive)
    p.add_argument('--policy-batch-size', type=positive)
    p.add_argument('--horizon', type=positive)
    p.add_argument('--save-interval', type=positive)
    p.add_argument('--log-interval', type=positive)
    p.add_argument('--no-compile', action='store_true', help='Run eager PyTorch; useful for compatibility smoke tests')
    p.add_argument('--dry-run', action='store_true', help='Validate and print config, files, versions; no simulation')
    return p


def merge_config(cfg, values):
    """递归覆盖上游 dataclass；未知字段报错，避免拼写错误悄悄改变实验。"""
    allowed = {f.name for f in fields(cfg)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f'Unknown {type(cfg).__name__} field: {key}')
        old = getattr(cfg, key)
        if is_dataclass(old):
            if not isinstance(value, dict):
                raise ValueError(f'{key} must be an object')
            merge_config(old, value)
        else:
            if isinstance(old, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def data_files(cfg):
    """固定动作列表顺序，并验证上游不会自动纠正的关节顺序和身体顺序。"""
    from g1_supermimic.core.envs.mdp.motion_library import discover_npz
    if cfg.data.library is not None:
        path = Path(cfg.data.library).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    directory = Path(cfg.data.npz_dir).expanduser()
    cfg.data.npz_dir = (directory if directory.is_absolute() else ROOT/directory).resolve()
    files = discover_npz(cfg.data.npz_dir, names=cfg.data.clips,
                         limit=cfg.data.limit, manifest=cfg.data.manifest)
    if not files:
        raise FileNotFoundError(f'No clips found under {cfg.data.npz_dir}')
    from g1_supermimic.core.data.motion_convert import G1_JOINT_NAMES
    from g1_supermimic.core.envs.tracking_env_cfg import TRACKED_BODIES
    from g1_supermimic.support.motion import inspect
    body_names = None
    for path in files:
        report = inspect(path, joint_names=G1_JOINT_NAMES, expected_fps=50.)
        names = report['body_names']
        if not set(TRACKED_BODIES) <= set(names):
            raise ValueError(f'Missing tracked body names: {path}')
        # 上游 full library 使用首个文件的 body 下标处理所有片段。
        if body_names is not None and names != body_names:
            raise ValueError(f'Mixed body order across motion clips: {path}')
        body_names = names
    return [p.resolve() for p in files]


def digest(path):
    # The simulator consumes these arrays, not ZIP timestamps or the converter's
    # machine-specific `source` metadata. Make equivalent motion files portable.
    path = Path(path)
    if path.suffix == '.npz':
        import numpy as np
        from .support.motion import REQUIRED
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = (*REQUIRED, 'fps', 'joint_names', 'body_names')
                if all(key in data for key in keys):
                    h = hashlib.sha256(b'godot-motion-arrays-v1\0')
                    for key in keys:
                        h.update(key.encode()+b'\0')
                        values = data[key]
                        if key.endswith('_names'):
                            payload = json.dumps(values.tolist(), ensure_ascii=False, separators=(',',':')).encode()
                        else:
                            values = np.asarray(values, dtype='<f4')
                            payload = json.dumps(list(values.shape)).encode()+values.tobytes(order='C')
                        h.update(len(payload).to_bytes(8,'little')+payload)
                    return h.hexdigest()
        except (ValueError, OSError):
            pass  # Generic byte fixtures/packed inputs retain ordinary file hashing.
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(files):
    from . import __version__
    from .runtime import configure_backend_compatibility
    versions={p:importlib.metadata.version(p) for p in ('torch','mjlab','mujoco','mujoco-warp','warp-lang')}
    return dict(implementation='Godot-SuperMimic',project_version=__version__,
                runtime_versions=versions,backend_compatibility=configure_backend_compatibility(),
                source_sha256=hashlib.sha256(''.join(digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))).encode()).hexdigest(),
                data_hash='motion_arrays_v1',data=[dict(path=str(p),sha256=digest(p)) for p in files])
