"""Fixed-start physical tracking errors, keeping global drift separate from local pose."""
import json
import math
from pathlib import Path


def tracking_errors(cmd):
    import torch
    from mjlab.utils.lab_api.math import quat_error_magnitude
    from g1_supermimic.core.envs.mdp.rewards import reanchor_reference

    def yaw(q):
        w, x, y, z = q.unbind(-1)
        return torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

    angle = yaw(cmd.anchor_quat_w) - yaw(cmd.robot_anchor_quat_w)
    yaw_error = torch.atan2(angle.sin(), angle.cos()).abs()
    aligned_pos, aligned_quat = reanchor_reference(
        cmd.body_pos_w, cmd.body_quat_w, cmd.anchor_pos_w, cmd.anchor_quat_w,
        cmd.robot_anchor_pos_w, cmd.robot_anchor_quat_w)
    return dict(
        anchor_xy_m=(cmd.anchor_pos_w[:, :2] - cmd.robot_anchor_pos_w[:, :2]).norm(dim=-1),
        anchor_yaw_deg=torch.rad2deg(yaw_error),
        anchor_rotation_deg=torch.rad2deg(quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)),
        body_world_m=(cmd.body_pos_w - cmd.robot_body_pos_w).norm(dim=-1).mean(-1),
        body_aligned_m=(aligned_pos - cmd.robot_body_pos_w).norm(dim=-1).mean(-1),
        body_aligned_rotation_deg=torch.rad2deg(quat_error_magnitude(aligned_quat, cmd.robot_body_quat_w)).mean(-1),
        joint_rmse_rad=(cmd.joint_pos - cmd.robot_joint_pos).square().mean(-1).sqrt(),
    )


def diagnose(args):
    if not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError('--seconds must be finite and nonnegative')
    from .inference import inference_context
    path, policy, cfg, files = inference_context(args)
    import torch
    from g1_supermimic.core.envs import SuperMimicEnvCfgArgs, make_env

    # One environment per clip; --envs does not change the sampling protocol.
    n = len(files)
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=n, future_frames=cfg.env.future_frames,
        episode_length_s=1.e5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode='start', record_transitions=False,
        seed=0, anchor_obs=cfg.env.anchor_obs), args.device)
    try:
        with torch.inference_mode():
            cmd = env.command_manager.get_term('motion')
            lib = cmd.library
            cmd.forced_frame[:] = lib.clip_start[torch.arange(n, device=args.device)]
            obs, _ = env.reset()
            active = torch.ones(n, dtype=torch.bool, device=args.device)
            steps = torch.zeros(n, dtype=torch.long, device=args.device)
            failed = torch.zeros_like(active)
            ret = torch.zeros(n, device=args.device)
            totals = {k: torch.zeros(n, device=args.device) for k in tracking_errors(cmd)}
            last = {k: v.clone() for k, v in totals.items()}
            first = {k: torch.full((n,), -1., device=args.device) for k in ('xy_gt_0p3_s', 'yaw_gt_15_s')}
            series = []
            step = 0
            max_steps = math.ceil(args.seconds / env.step_dt) if args.seconds else None
            names = list(env.reward_manager.active_terms)
            term_sum = torch.zeros(n, len(names), device=args.device)
            while bool(active.any()):
                values = tracking_errors(cmd)
                for key, value in values.items():
                    totals[key] += torch.where(active, value, 0.)
                    last[key] = torch.where(active, value, last[key])
                for key, exceeded in (
                    ('xy_gt_0p3_s', values['anchor_xy_m'] > .3),
                    ('yaw_gt_15_s', values['anchor_yaw_deg'] > 15.)):
                    first[key] = torch.where(active & (first[key] < 0) & exceeded, step * env.step_dt, first[key])
                if step % max(1, round(1 / env.step_dt)) == 0:
                    ids = active.nonzero().squeeze(-1)
                    series.append(dict(seconds=step*env.step_dt, active=int(active.sum()),
                                       **{k: float(v[ids].mean()) for k, v in values.items()}))
                obs, reward, term, timeout, _ = env.step(policy(obs['policy'], obs['goal']))
                ret += torch.where(active, reward, 0.)
                manager = env.reward_manager
                terms = manager.term_rewards if hasattr(manager, 'term_rewards') else manager._step_reward * (env.step_dt if manager._scale_by_dt else 1.)
                term_sum += terms * active[:, None]
                steps += active.long()
                failed |= term & active
                active &= ~(term | timeout)
                step += 1
                if step % 1000 == 0:
                    print(f'DIAGNOSE steps={step} active={int(active.sum())}/{n}', flush=True)
                if max_steps and step >= max_steps:
                    break
            count = steps.clamp_min(1)
            rows = []
            for i in range(n):
                rows.append(dict(name=lib.clips[i].name, frames=lib.clips[i].length,
                    steps=int(steps[i]), failed=bool(failed[i]), censored=bool(active[i]),
                    norm_return=float(ret[i] / (.1 * count[i])),
                    full_clip_norm_return=float(ret[i] / (.1 * lib.clips[i].length)),
                    mean={k: float(v[i] / count[i]) for k, v in totals.items()},
                    final={k: float(v[i]) for k, v in last.items()},
                    **{k: float(v[i]) for k, v in first.items()}))
            summary = dict(clips=n, failed=int(failed.sum()), censored=int(active.sum()),
                mean_norm_return=sum(r['norm_return'] for r in rows)/n,
                mean_full_clip_norm_return=sum(r['full_clip_norm_return'] for r in rows)/n,
                step_weighted_errors={k: float(v.sum() / count.sum()) for k, v in totals.items()},
                terms={key: float(term_sum[:, i].sum() / count.sum()) for i, key in enumerate(names)})
            for key, value in first.items():
                valid = value[value >= 0]
                summary[key] = dict(clips_crossed=int(valid.numel()), median=float(valid.median()) if valid.numel() else None)
            report = dict(checkpoint=str(path), protocol='deterministic first episode from frame zero, no noise/DR; errors before step, rewards after step',
                reference_scope='unseen_motion' if getattr(args, 'allow_new_motion', False) else 'checkpoint_data',
                seconds_limit=args.seconds, step_dt=env.step_dt, summary=summary, clips=rows, series=series)
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(f'DIAGNOSIS={output}\n{json.dumps(summary, ensure_ascii=False)}', flush=True)
    finally:
        env.close()
