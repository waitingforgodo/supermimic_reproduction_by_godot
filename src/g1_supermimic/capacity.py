"""A lower-bound CUDA allocation probe at the exact configured batch/horizon.

Replay and the policy graphs alone must fit before the simulator, world model,
critic or optimizer can be added. The upstream policy itself is not compiled by
the native trainer; only the world-model/analytic step is compiled.
"""
import gc


def probe(cfg):
    import torch
    from g1_supermimic.core.data.replay_buffer import ReplayBuffer
    from .variants import make_policy

    device = torch.device(cfg.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        return dict(passed=False, reason='The upstream training capacity probe requires an available CUDA device')
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = dict(passed=False, scope='Necessary lower bound: replay plus uncompiled policy forward graphs; excludes simulator, world model, critic and optimizers',
                  device=str(device), gpu_name=torch.cuda.get_device_name(device),
                  gpu_total_bytes=total, gpu_free_bytes_before=free,
                  num_envs=cfg.env.num_envs, replay_capacity=cfg.env.replay_capacity,
                  policy_batch_size=cfg.policy_trainer.batch_size,
                  horizon=cfg.policy_trainer.horizon, completed_policy_steps=0)
    replay = policy = observations = goals = output = None
    outputs = []
    stage = 'replay allocation'
    try:
        replay = ReplayBuffer(cfg.env.num_envs, cfg.env.replay_capacity, 71, 29, 33, device)
        result['replay_allocated_bytes'] = torch.cuda.memory_allocated(device)
        stage = 'policy initialization'
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]):
            policy = make_policy(cfg).to(device)
            if getattr(getattr(cfg, 'performance', None), 'precision', 'fp32') == 'bf16':
                from .performance import autocast_network
                autocast_network(policy)
        observations = torch.zeros(cfg.policy_trainer.batch_size, cfg.policy.history * 93,
                                   device=device, requires_grad=True)
        goals = torch.zeros(cfg.policy_trainer.batch_size, cfg.policy.goal_dim, device=device)
        stage = 'policy horizon graphs'
        # Match the forward-only cast-cache lifetime of the imagined rollout.
        with torch.autocast(device_type=device.type, enabled=False):
            for step in range(cfg.policy_trainer.horizon):
                output = policy(observations, goals)
                outputs.append(output)
                result['completed_policy_steps'] = step + 1
        torch.cuda.synchronize(device)
        result['allocation_succeeded'] = True
        # Some CUDA/WSL drivers allow allocations beyond physical VRAM. That
        # is not evidence that a full training job fits in device memory.
        needed = torch.cuda.max_memory_allocated(device) - allocated_before
        result['required_resident_bytes_lower_bound'] = needed
        result['passed'] = needed <= free
        result['reason'] = ('Lower-bound allocations fit in free physical VRAM; full training may still require more memory'
                            if result['passed'] else
                            'Replay and policy graphs alone exceed free physical VRAM; allocation succeeded through device-memory oversubscription')
    except torch.cuda.OutOfMemoryError as exc:
        result['reason'] = 'CUDA out of memory at the exact configured dimensions; full native training cannot fit on this device'
        result['error'] = str(exc)
        result['failed_stage'] = stage
    finally:
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        outputs.clear()
        replay = policy = observations = goals = output = None
        gc.collect()
        torch.cuda.empty_cache()
    return result
