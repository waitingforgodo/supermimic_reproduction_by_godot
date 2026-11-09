"""Bounded resume smoke test; writes an independent run and audit snapshot."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('models/skills/best.pt'))
    parser.add_argument('--steps',type=int,default=48)
    parser.add_argument('--name',default='release_resume48')
    args=parser.parse_args()
    import torch
    from g1_supermimic.cli import parser as command_parser
    from g1_supermimic.training import train
    from g1_supermimic.retention import BestRetention
    from g1_supermimic.safety import require_finite
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    before=source['num_updates']
    options=command_parser().parse_args(['resume','--checkpoint',str(args.checkpoint),
        '--envs','16','--world-batch-size','16','--world-batches-per-update','1',
        '--policy-batch-size','16','--horizon','8','--iterations','20000',
        '--stop-after-steps',str(args.steps),'--save-interval','1000','--no-compile',
        '--run-root','outputs/release_validation/runs','--name',args.name])
    original=BestRetention.consider
    snapshots=[]
    def consider(self):
        if self.candidate.is_file():
            path=self.folder/'validation_snapshot.pt'
            shutil.copyfile(self.candidate,path)
            snapshots.append(path)
        return original(self)
    with patch.object(BestRetention,'consider',consider):
        train(options)
    assert snapshots,'No completed training snapshot'
    final=torch.load(snapshots[-1],map_location='cpu',weights_only=False)
    require_finite(final,'resume validation snapshot')
    assert final['iteration']==source['iteration']+args.steps
    assert final['num_updates']>before
    def steps(checkpoint,key):
        return sorted({float(value['step']) for value in checkpoint[key]['state'].values() if 'step' in value})
    optimizer={}
    for key in ('wm_opt','policy_opt','critic_opt'):
        old,new=steps(source,key),steps(final,key)
        assert old and new and min(new)>min(old),(key,old,new)
        optimizer[key]=dict(before=old,after=new)
    result=dict(status='passed',scope='reduced-size GPU resume smoke; not a production throughput benchmark',
        envs=16,horizon=8,batch_size=16,start_iteration=source['iteration'],
        final_iteration=final['iteration'],start_updates=before,final_updates=final['num_updates'],
        optimizer_steps=optimizer,checkpoint_tensors_finite=True)
    output=Path('results/gpu_resume_validation.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
