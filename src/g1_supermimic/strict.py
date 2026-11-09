"""Compare effective training settings with the pinned upstream dataclasses."""
from dataclasses import asdict
import json
from pathlib import Path

from .configuration import digest

LAFAN_DATASET = 'lvhaidong/LAFAN1_Retargeting_Dataset'
from .paths import ROOT
LOCAL_LAFAN = ROOT / 'data/lafan1'


def config_differences(actual, expected, prefix=''):
    differences = {}
    for key in actual.keys() | expected.keys():
        name = f'{prefix}.{key}' if prefix else key
        left, right = actual.get(key), expected.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            differences.update(config_differences(left, right, name))
        elif left != right or type(left) is not type(right):
            differences[name] = dict(upstream=right, effective=left)
    return differences


def training_contract(cfg, upstream_defaults, files, *, strict=True):
    actual, expected = asdict(cfg), asdict(upstream_defaults)
    # Machine-specific locations and the selected CUDA device do not change
    # training hyperparameters. The dataset itself is checked below by hashes.
    for key in ('run_name', 'log_root', 'device', 'resume'):
        actual.pop(key, None)
        expected.pop(key, None)
    actual['data'].pop('npz_dir', None)
    expected['data'].pop('npz_dir', None)
    differences = config_differences(actual, expected)
    if strict and differences:
        fields = ', '.join(f'{key}: {row["effective"]!r} (upstream {row["upstream"]!r})'
                           for key, row in sorted(differences.items()))
        raise ValueError('Strict upstream configuration differs: ' + fields +
                         '. Remove overrides, or explicitly use --allow-config-changes for a custom experiment.')
    data = None
    if strict:
        folder = Path(cfg.data.npz_dir).expanduser().resolve()
        metadata = folder / 'dataset_manifest.json'
        if not metadata.is_file():
            raise FileNotFoundError(f'{metadata}; run g1.sh prepare for the pinned 40-clip LAFAN1 dataset')
        data = json.loads(metadata.read_text())
        outputs = data.get('outputs', {})
        if (data.get('dataset') != LAFAN_DATASET or data.get('clips') != 40
                or len(outputs) != 40 or len(files) != 40
                or {p.name for p in files} != set(outputs)):
            raise ValueError('Strict upstream LAFAN1 training requires exactly the pinned 40 clips without filtering')
        for path in files:
            if digest(path) != outputs[path.name]:
                raise ValueError(f'Strict upstream LAFAN1 data differs from its conversion manifest: {path}')
        data = {key: data.get(key) for key in ('dataset', 'revision', 'clips', 'frames', 'duration_hours')}
    return dict(mode='strict_upstream' if strict else 'custom',
                upstream_parameters_match=not differences,
                parameter_differences=differences, dataset=data,
                path_remapping={'data.npz_dir': str(Path(cfg.data.npz_dir).resolve())},
                scope='Pinned repository training settings; runtime-lock equivalence is reported separately')
