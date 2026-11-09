"""Record closed-loop checkpoint policies; render recordings as MP4 and GIF."""
from pathlib import Path
import json
import os


def add_parser(sub):
    parser=sub.add_parser('record',help='Record actual policy tracking and save MP4/GIF demonstrations')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--npz-dir',type=Path)
    parser.add_argument('--clips',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seconds',type=float,default=0.,help='0: complete selected clips; positive: demonstration cap')
    parser.add_argument('--fps',type=int,choices=(10,25,50),default=25)
    parser.add_argument('--width',type=int,default=960)
    parser.add_argument('--height',type=int,default=540)
    parser.add_argument('--no-render',action='store_true')
    return parser


def record(args):
    # Set before importing MuJoCo via the runtime, so a headless run is portable.
    os.environ.setdefault('MUJOCO_GL','osmesa')
    os.environ.setdefault('LP_NUM_THREADS','2')
    if args.seconds < 0:raise ValueError('--seconds must be nonnegative')
    if args.width < 64 or args.height < 64 or args.width%2 or args.height%2:
        raise ValueError('Video dimensions must be positive even numbers >=64')
    from argparse import Namespace
    from .inference import inference_context
    from .configuration import digest
    from .support.contract import sha256,write_json
    path,policy,cfg,all_files=inference_context(Namespace(checkpoint=args.checkpoint,device=args.device,
        npz_dir=args.npz_dir,motion=None,allow_new_motion=False,policy_source='auto'))
    by_name={p.stem:p for p in all_files}
    if len(set(args.clips))!=len(args.clips) or not set(args.clips)<=by_name.keys():
        raise ValueError(f'Choose unique clips from {sorted(by_name)}')
    files=[by_name[name] for name in args.clips]
    import numpy as np
    import torch
    from .core.envs import SuperMimicEnvCfgArgs,make_env
    from .core.envs.state_io import StateIO
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(p) for p in files],num_envs=len(files),
        future_frames=cfg.env.future_frames,anchor_obs=cfg.env.anchor_obs,episode_length_s=1.e5,
        action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,
        sampling_mode='start',record_transitions=False,seed=0),device=args.device)
    robots=[[] for _ in files]
    references=[[] for _ in files]
    try:
        command=env.command_manager.get_term('motion')
        lib=command.library
        io=StateIO(env)
        command.forced_frame[:]=lib.clip_start
        obs,_=env.reset()
        command.forced_frame[:]=lib.clip_start
        lengths=torch.tensor([clip.length for clip in lib.clips],device=args.device)
        caps=lengths.clone()
        if args.seconds>0:caps=caps.clamp_max(max(1,round(args.seconds/env.step_dt)))
        active=torch.ones(len(files),dtype=torch.bool,device=args.device)
        terminated=torch.zeros_like(active)
        completed=torch.zeros_like(active)
        counts=torch.zeros(len(files),dtype=torch.long,device=args.device)
        reward_sum=torch.zeros(len(files),device=args.device)
        with torch.inference_mode():
            while bool(active.any()):
                robot=io.read()[:,:36].cpu().numpy()
                reference=lib.ref_qpos_qvel(command.frame)[0].cpu().numpy()
                active_ids=active.nonzero().flatten().tolist()
                for i in active_ids:
                    robots[i].append(robot[i].copy())
                    references[i].append(reference[i].copy())
                action=policy(obs['policy'],obs['goal'])
                if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite recording action')
                obs,reward,term,tout,_=env.step(action)
                reward_sum+=reward*active
                counts+=active.long()
                terminated|=term&active
                completed|=tout&active
                active&=~(term|tout| (counts>=caps))
                if not torch.isfinite(reward).all():raise FloatingPointError('Nonfinite recording reward')
        nframes=[len(frames) for frames in robots]
        total=max(nframes)
        def padded(clips):
            return np.stack([np.concatenate((np.asarray(frames),np.repeat(np.asarray(frames[-1])[None],total-len(frames),axis=0)))
                             for frames in clips])
        output=args.output.expanduser().resolve()
        output.mkdir(parents=True,exist_ok=True)
        trajectory=output/'rollouts.npz'
        np.savez_compressed(trajectory,names=np.array(args.clips),robot_qpos=padded(robots),ref_qpos=padded(references),
            lengths=np.array(nframes),norm_return=(reward_sum/(.1*counts.clamp_min(1))).cpu().numpy())
        metadata=dict(checkpoint=str(path),checkpoint_sha256=sha256(path),policy_source=cfg._policy_source,
            recording='actual deterministic closed-loop policy in MuJoCo Warp; simulation, not hardware',
            seed=0,control_dt=env.step_dt,history=cfg.policy.history,demonstration_cap_seconds=args.seconds,
            clips=[dict(name=name,frames=nframes[i],reference_frames=int(lengths[i]),
                        failure=bool(terminated[i]),clip_end=bool(completed[i]),
                        truncated_for_demo=nframes[i]<int(lengths[i]) and not bool(terminated[i]),
                        motion_content_sha256=digest(files[i])) for i,name in enumerate(args.clips)])
        write_json(output/'recording.json',metadata)
        print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
    finally:
        env.close()
    if not args.no_render:
        from .rendering import main as render
        render(trajectory,output,fps=args.fps,width=args.width,height=args.height,
               concat=output/'showcase.mp4')
        gif(output/'showcase.mp4',output/'showcase.gif',seconds=12)
    return output


def gif(video,output,seconds=12,width=560):
    import imageio_ffmpeg
    import subprocess
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-threads','2',
        '-i',str(video),'-t',str(seconds),'-vf',
        f'fps=10,scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer',
        '-filter_complex_threads','1',str(output)]
    subprocess.run(command,check=True)
