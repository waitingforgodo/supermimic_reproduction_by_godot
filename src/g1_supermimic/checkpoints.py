"""上游 checkpoint 格式、动作哈希和本方法 latest 选择。"""
import json
from pathlib import Path
from .paths import RUNS
from .configuration import digest


def resolve_checkpoint(value):
    """latest selects the newest run; best ranks validated, matching motion sets."""
    if str(value) in ('latest', 'best'):
        candidates = [p for folder in RUNS.glob('*') if folder.is_dir()
                      for p in [folder / 'best.pt' if (folder / 'best.pt').is_file() else folder / 'ckpt_latest.pt']
                      if p.is_file()]
        if not candidates:
            raise FileNotFoundError('No Godot-SuperMimic checkpoint; train with g1.sh train')
        latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        if str(value) == 'latest' or len(candidates) == 1:
            return latest
        from .retention import benchmark_score
        records = {}
        for candidate in candidates:
            try:
                index = json.loads((candidate.parent / 'best.json').read_text())
                report = index['report']
                data = report['data']
                score = benchmark_score(report, data)
                signature = tuple((Path(row['path']).name, row['sha256']) for row in data)
                records[candidate] = signature, score
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if latest not in records:
            raise ValueError('Cannot rank best without a valid best.json for the latest run; use latest or an explicit checkpoint')
        signature = records[latest][0]
        eligible = [p for p, record in records.items() if record[0] == signature]
        return max(eligible, key=lambda p: (records[p][1], p.stat().st_mtime_ns))
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / ('best.pt' if (path / 'best.pt').is_file() else 'ckpt_latest.pt')
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = path.parent / 'run_manifest.json'
    if manifest.exists():
        implementation = json.loads(manifest.read_text()).get('implementation')
        if implementation != 'Godot-SuperMimic':
            raise ValueError('Unsupported checkpoint implementation; only Godot-SuperMimic is supported')
    return path


def load_checkpoint(path):
    """只加载可信的本地/上游权重；先检查结构，再交给上游网络恢复。"""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or not {'wm', 'policy', 'critic', 'config', 'iteration'} <= ck.keys():
        raise ValueError('Expected a Godot-SuperMimic checkpoint')
    from .safety import require_finite
    require_finite(ck, f'checkpoint {Path(path).name}')
    return ck


def check_data_contract(checkpoint, files):
    """本地产生的模型须沿用相同动作及顺序，避免把换动作算成续训收益。"""
    manifest = Path(checkpoint).parent / 'run_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        actual = [digest(p) for p in files]
        if actual != [row['sha256'] for row in saved['data']]:
            raise ValueError('Reference motion hashes/order differ from checkpoint manifest')
