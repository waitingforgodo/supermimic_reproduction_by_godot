"""真实仿真采集与世界模型/BPTT 训练；优化由上游 Trainer 完成。"""
from datetime import datetime
import json
from pathlib import Path
from .paths import ROOT, RUNS
from .runtime import initialize
from .configuration import merge_config, data_files, provenance
from .checkpoints import resolve_checkpoint, load_checkpoint, check_data_contract
from g1_supermimic.support.contract import write_json


def configure(args):
    # Programmatic callers must inherit the same resume config as cli.main().
    if getattr(args, 'checkpoint', None) is not None:
        args.resume = args.checkpoint
    init_from = getattr(args, 'init_from', None)
    if init_from is not None and args.resume is not None:
        raise ValueError('--init-from cannot be combined with --resume or --checkpoint')
    if init_from is not None and not (args.motion or args.npz_dir):
        raise ValueError('--init-from requires an explicit target --motion or --npz-dir')
    initialize()
    from .variants import TrainConfig, prepare_variant, REFERENCE
    cfg = TrainConfig()
    from .strict import LOCAL_LAFAN, training_contract
    cfg.data.npz_dir = LOCAL_LAFAN
    resumed = None
    initial = None
    if args.resume:
        args.resume = resolve_checkpoint(args.resume)
        resumed = load_checkpoint(args.resume)
        merge_config(cfg, resumed['config'])
    elif init_from is not None:
        init_from = resolve_checkpoint(init_from)
        initial = load_checkpoint(init_from)
        merge_config(cfg, initial['config'])
    elif not args.strict_upstream:
        profile = Path(__file__).parent / 'profiles/lafan1_4096_h16_eager.json'
        merge_config(cfg, json.loads(profile.read_text()))
        cfg.data.npz_dir = LOCAL_LAFAN
    if args.config:
        merge_config(cfg, json.loads(args.config.read_text()))
    if args.motion:
        path = args.motion.expanduser().resolve()
        cfg.data.npz_dir, cfg.data.clips = path.parent, [path.stem]
        cfg.data.library, cfg.data.manifest, cfg.data.limit = None, None, None
    if args.npz_dir:
        cfg.data.npz_dir = args.npz_dir.expanduser().resolve()
        cfg.data.library, cfg.data.manifest, cfg.data.limit = None, None, None
        cfg.data.clips = None
    if args.clips:
        cfg.data.clips = args.clips
    overrides = (
        (cfg.env, 'num_envs', args.envs), (cfg.env, 'replay_capacity', args.replay_capacity),
        (cfg.schedule, 'total_iterations', args.iterations),
        (cfg.schedule, 'checkpoint_interval', args.save_interval),
        (cfg.schedule, 'log_interval', args.log_interval),
        (cfg.wm_trainer, 'batch_size', args.world_batch_size),
        (cfg.wm_trainer, 'batches_per_update', args.world_batches_per_update),
        (cfg.policy_trainer, 'batch_size', args.policy_batch_size),
        (cfg.policy_trainer, 'horizon', args.horizon),
        (cfg, 'device', args.device), (cfg, 'seed', args.seed),
    )
    for section, key, value in overrides:
        if value is not None:
            setattr(section, key, value)
    if args.no_compile:
        cfg.wm_trainer.compile = cfg.policy_trainer.compile = None
    from .learning_rates import base_learning_rates
    learning_rates = base_learning_rates(cfg)
    from .retention import validate_confirmations
    validate_confirmations(cfg.checkpoint_confirmations)
    cfg.resume = args.resume
    cfg.log_root = args.run_root.expanduser().resolve() if getattr(args, 'run_root', None) else RUNS
    cfg.run_name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S_%f_reproduced')
    if Path(cfg.run_name).name != cfg.run_name or cfg.run_name in ('.', '..'):
        raise ValueError('--name must be a single directory name')
    # 固定历史长度/目标帧数与 Policy 的默认形状一致，不开放不完整的维度修改。
    if cfg.policy.history != 10 or cfg.env.future_frames != 10 or cfg.policy_trainer.future_frames != 10 or cfg.policy_trainer.history != 10:
        raise ValueError('This bridge requires history=10 and future_frames=10')
    if cfg.env.replay_capacity <= max(cfg.wm_trainer.window, 11):
        raise ValueError('Replay capacity must exceed both world window and history+1')
    if cfg.schedule.warmup_iterations <= cfg.wm_trainer.window:
        raise ValueError('Warmup must exceed the world-model window')
    if cfg.policy_trainer.batch_size * cfg.policy_trainer.horizon < cfg.policy_trainer.critic_minibatches:
        raise ValueError('Critic minibatches would be empty')
    if cfg.data.library is not None:
        cfg.env.compact_library = True
    files = data_files(cfg)
    prepare_variant(cfg, resumed if resumed is not None else initial)
    initialization = None
    if initial is not None:
        from .initialization import prepare_initialization
        initialization = prepare_initialization(cfg, initial, init_from)
        del initial
    from .ema import prepare_ema
    from .performance import prepare_performance
    ema = prepare_ema(cfg, resumed)
    performance = prepare_performance(cfg)
    from .style_amp import prepare_style_amp
    style_amp = prepare_style_amp(cfg, resumed, files)
    from .sampling import prepare_sampling
    sampling = prepare_sampling(cfg, resumed, num_clips=len(files) if cfg.data.library is None else None)
    from .stabilization import prepare_stabilization
    stability = prepare_stabilization(cfg, resumed, files)
    from .reward_alignment import alignment_manifest
    alignment = alignment_manifest(cfg)
    if args.resume:
        check_data_contract(args.resume, files)
        if cfg.schedule.total_iterations <= resumed['iteration']:
            raise ValueError('--iterations must exceed the resumed env-step count')
    contract = training_contract(cfg, TrainConfig(), files,
                                 strict=args.strict_upstream)
    manifest = provenance(files)
    manifest['performance'] = performance
    if style_amp is not None:
        manifest['style_amp'] = style_amp
    if initialization is not None:
        manifest['initialization'] = initialization
    if ema is not None:
        manifest['policy_ema'] = ema
    if sampling is not None:
        manifest['reset_sampling'] = sampling
    if stability is not None:
        manifest['policy_stabilization'] = stability
    if alignment is not None:
        manifest['world_model_reward_alignment'] = alignment
    manifest['iteration_unit'] = 'vectorized_environment_control_step'
    manifest['policy_observations'] = dict(
        history_frames=cfg.policy.history, proprioception_per_frame=93,
        actor_goal_dim=cfg.policy.goal_dim - (3 if cfg.optimizations.motion_anchor_ori_b else 0),
        anchor_orientation_feedback=bool(cfg.env.anchor_obs),
        anchor_position_feedback=bool(cfg.env.anchor_obs and not cfg.optimizations.motion_anchor_ori_b),
        normalization_updates='real collection only; inference and imagination leave statistics frozen')
    manifest['lr_schedule'] = dict(
        unit='vectorized_environment_control_step', total_steps=cfg.schedule.total_iterations,
        base_learning_rates=learning_rates, independent_critic_learning_rate=True,
        cosine_decay_steps=cfg.schedule.total_iterations * cfg.schedule.lr_decay_fraction,
        final_ratio=cfg.schedule.lr_final_ratio,
        warmup_steps=cfg.schedule.warmup_iterations,
        updates_per_control_step=cfg.schedule.updates_per_env_step,
        short_run_stop_does_not_rescale_schedule=True)
    if any(vars(cfg.optimizations).values()):
        manifest['optimization_reference'] = REFERENCE
        manifest['optimization_options'] = vars(cfg.optimizations).copy()
    manifest['training_contract'] = contract
    manifest['configuration_base'] = ('checkpoint_weights' if initialization is not None else
                                      'checkpoint' if resumed is not None else
                                      'strict_upstream' if args.strict_upstream else
                                      'lafan1_4096_h16_eager')
    manifest['numerical_guard'] = (
        'finite actions, gradients, update metrics and atomic finite-only checkpoints; '
        + ('upstream BPTT with explicitly recorded reward alignment/reference penalty'
           if stability is not None or alignment is not None else 'upstream losses/updates unchanged'))
    from .retention import POLICY
    manifest['checkpoint_retention'] = dict(POLICY, interval=cfg.schedule.checkpoint_interval,
        candidate_confirmations=cfg.checkpoint_confirmations,
        confirmation_score='worst lexicographic score; repeat only a prospective improvement')
    if args.resume:
        manifest['source_checkpoint'] = str(args.resume)
        from .configuration import digest
        manifest['source_checkpoint_sha256'] = digest(args.resume)
    return cfg, manifest


def capacity_report(cfg, manifest):
    from dataclasses import asdict
    from .capacity import probe
    result = probe(cfg)
    output = ROOT / 'outputs/supermimic/checks' / f'{datetime.now():%Y%m%d_%H%M%S_%f}_capacity.json'
    report = dict(config=asdict(cfg), manifest=manifest, capacity=result)
    write_json(output, report)
    print(f'CAPACITY_REPORT={output}\n{json.dumps(result, indent=2)}', flush=True)
    return result, output


def preflight(args):
    from dataclasses import asdict
    cfg, manifest = configure(args)
    if args.dry_run:
        print(json.dumps(dict(config=asdict(cfg), manifest=manifest), ensure_ascii=False, indent=2, default=str))
        return
    from g1_supermimic.support.locking import training_lock
    with training_lock():
        result, output = capacity_report(cfg, manifest)
    if not result['passed']:
        raise ValueError(f'Configured native training exceeds available GPU capacity; parameters were not reduced. See {output}')


def train(args):
    from dataclasses import asdict
    cfg, manifest = configure(args)
    log_dir = Path(cfg.log_root) / cfg.run_name
    if args.dry_run:
        print(json.dumps(dict(config=asdict(cfg), manifest=manifest), ensure_ascii=False, indent=2, default=str))
        return
    if log_dir.exists():
        raise FileExistsError(f'Existing run is preserved: {log_dir}; choose a new --name')
    from g1_supermimic.support.locking import training_lock
    with training_lock():
        result, output = capacity_report(cfg, manifest)
        if not result['passed']:
            raise ValueError(f'Configured native training exceeds GPU capacity; no run was started and no parameters were reduced. See {output}')
        manifest['capacity_report'] = str(output)
        if getattr(args, 'stop_after_steps', None) is not None:
            manifest['short_run'] = dict(additional_control_steps=args.stop_after_steps,
                                         lr_schedule_unchanged=True)
        log_dir.mkdir(parents=True, exist_ok=False)
        write_json(log_dir / 'run_manifest.json', manifest)
        print(f'RUN={log_dir}\nTENSORBOARD_LOGDIR={log_dir / "tb"}', flush=True)
        from .variants import make_trainer
        from g1_supermimic.support.telemetry import ResourceSampler
        from .safety import guard_training
        from .retention import retain_best
        from .short_run import stop_after_steps, ShortRunComplete
        import time
        import torch
        sampler = ResourceSampler()
        sampler.start()
        started = time.perf_counter()
        tr = None
        status, error = 'failed', None
        stage = {'name': 'trainer initialization'}
        try:
            torch.cuda.reset_peak_memory_stats(cfg.device)
            tr = make_trainer(cfg)
            with guard_training(tr) as stage:
                with retain_best(tr, manifest), stop_after_steps(tr, getattr(args, 'stop_after_steps', None)):
                    try:
                        tr.run()
                        status = 'completed' if tr.iteration >= cfg.schedule.total_iterations else 'time_limited'
                    except ShortRunComplete:
                        tr.save()
                        if tr.writer:
                            tr.writer.flush()
                        status = 'short_run_completed'
                        print('Short run completed; evaluated the final candidate and retained best.pt', flush=True)
                    except KeyboardInterrupt:
                        tr.save()
                        status = 'interrupted'
                        print('Interrupted; evaluated the final candidate and retained best.pt', flush=True)
        except FloatingPointError as exc:
            status, error = 'numerical_failure', str(exc)
            raise
        except torch.cuda.OutOfMemoryError as exc:
            status, error = 'out_of_memory', str(exc)
            raise ValueError('Full native training ran out of GPU memory; original settings were retained') from exc
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            resources = sampler.finish()
            resources.update(status=status, error=error, elapsed_seconds=time.perf_counter() - started,
                             last_training_stage=stage['name'],
                             torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg.device),
                             torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(cfg.device),
                             num_envs=cfg.env.num_envs, requested_environment_steps=cfg.schedule.total_iterations,
                             completed_environment_steps=tr.iteration if tr else 0,
                             completed_updates=tr.num_updates if tr else 0)
            if tr is not None and hasattr(tr.buffer, '_replay_cache_stats'):
                resources['replay_cache'] = tr.buffer._replay_cache_stats
            if tr is not None and hasattr(tr, 'style_amp'):
                resources['style_amp_updates'] = tr.style_amp.updates
            write_json(log_dir / 'resources.json', resources)
            if tr is not None:
                if tr.writer:
                    tr.writer.close()
                tr.env.close()
