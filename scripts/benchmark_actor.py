"""Actor-only latency; synthetic inputs, no simulation or robot communication."""
import argparse
from pathlib import Path
import json
import statistics
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--output',type=Path,default=Path('outputs/actor_latency.json'))
    args=parser.parse_args()
    if args.threads<1 or args.samples<2 or args.warmup<0:parser.error('Invalid benchmark counts')
    import torch
    from g1_supermimic.runtime import initialize
    from g1_supermimic.variants import TrainConfig,prepare_variant,make_policy
    from g1_supermimic.configuration import merge_config
    from g1_supermimic.support.contract import sha256,write_json
    initialize()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=merge_config(TrainConfig(),saved['config'])
    prepare_variant(cfg)
    actor=make_policy(cfg).to(args.device).eval()
    actor.load_state_dict(saved['policy'])
    del saved
    observations=torch.randn(1,93*cfg.policy.history,device=args.device)*.01
    goal=torch.zeros(1,cfg.policy.goal_dim,device=args.device)
    for frame in range(cfg.env.future_frames):
        goal[:,frame*68+60]=-1.
        goal[:,frame*68+67]=.78
    if cfg.env.anchor_obs:
        goal[:,-6:]=torch.tensor([1.,0.,0.,1.,0.,0.],device=args.device)
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize()
    timings=[]
    with torch.inference_mode():
        for _ in range(args.warmup):actor(observations,goal)
        sync()
        for _ in range(args.samples):
            start=time.perf_counter()
            output=actor(observations,goal)
            sync()
            timings.append(1000*(time.perf_counter()-start))
    assert output.shape==(1,29) and torch.isfinite(output).all()
    ordered=sorted(timings)
    result=dict(device=args.device,torch_version=torch.__version__,cpu_threads=args.threads,
        batch_size=1,samples=args.samples,warmup_excluded=args.warmup,
        median_ms=statistics.median(timings),p95_ms=ordered[min(len(ordered)-1,int(.95*len(ordered)))],
        checkpoint_sha256=sha256(args.checkpoint),history_dim=observations.shape[-1],
        transport_goal_dim=goal.shape[-1],effective_goal_dim=actor.encoders['motion'].scale.numel(),
        scope='Actor forward only, synthetic inputs, shared workstation; no physics/sensors/DDS/actuator deadline claim')
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
