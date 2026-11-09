"""Global policy-parameter backtracking; preserves the learned dynamics/sampler."""
from copy import copy
from pathlib import Path
import json
import math


def blend_policy(base, candidate, parameter_names, alpha):
    import torch
    if any(checkpoint['config'].get('ema', {}).get('decay', 0) for checkpoint in (base, candidate)):
        raise ValueError('EMA checkpoints cannot be used for two-point policy blending')
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be a finite number strictly between 0 and 1')
    a, b = base['policy'], candidate['policy']
    names = set(parameter_names)
    if a.keys() != b.keys() or names - set(b):
        raise ValueError('Policy architectures differ')
    if base['config'].get('optimizations', {}) != candidate['config'].get('optimizations', {}):
        raise ValueError('Policy feature semantics differ')
    for key in b:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            raise ValueError(f'Incompatible policy tensor: {key}')
        if key not in names and not torch.equal(a[key], b[key]):
            raise ValueError(f'Normalization/scales differ: {key}')
    result = copy(candidate)
    result['policy'] = {key: (1-alpha)*a[key] + alpha*b[key] if key in names else b[key] for key in b}
    result['policy_opt'] = dict(candidate['policy_opt'], state={})
    result.pop('selection', None)  # A derived policy requires its own physical evaluation.
    result['policy_initialization'] = dict(method='global_parameter_backtracking', alpha=alpha,
        policy_optimizer='reset Adam moments; other optimizers unchanged',
        training_iteration_is_parent=True)
    from .safety import require_finite
    require_finite(result['policy'], 'blended policy')
    return result


def add_arguments(p):
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    return p


def initialize_checkpoint(args):
    if args.output_dir.exists():
        raise FileExistsError('Output directory already exists; choose a new path')
    from g1_supermimic.runtime import initialize
    initialize()
    import torch
    from g1_supermimic.checkpoints import resolve_checkpoint, load_checkpoint, check_data_contract
    from g1_supermimic.configuration import merge_config, data_files, provenance, digest
    from g1_supermimic.variants import TrainConfig, prepare_variant, make_policy
    base_path, candidate_path = map(resolve_checkpoint, (args.base, args.candidate))
    base, candidate = map(load_checkpoint, (base_path, candidate_path))
    cfg = merge_config(TrainConfig(), candidate['config'])
    prepare_variant(cfg)
    policy = make_policy(cfg)
    policy.load_state_dict(candidate['policy'])
    files = data_files(cfg)
    for path in (base_path, candidate_path):
        check_data_contract(path, files)
    result = blend_policy(base, candidate, dict(policy.named_parameters()), args.alpha)
    result['policy_initialization'].update(
        base_checkpoint=str(base_path), base_sha256=digest(base_path),
        candidate_checkpoint=str(candidate_path), candidate_sha256=digest(candidate_path))
    manifest = provenance(files)
    manifest['policy_initialization'] = result['policy_initialization']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(result, args.output_dir/'initial.pt')
    for name, value in (('run_manifest.json', manifest), ('config.json', result['config']),
                        ('initialization.json', result['policy_initialization'])):
        (args.output_dir/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n')
    print(f'UNEVALUATED_CHECKPOINT={args.output_dir / "initial.pt"}')
