"""SuperMimic 功能入口：train/resume/play/evaluate/deploy。"""
import argparse
from pathlib import Path
from .configuration import train_parser, positive
from .inference import inference_parser


def parser():
    p = argparse.ArgumentParser(prog='g1.sh', description='SuperMimic：世界模型/BPTT，直接 checkpoint 仿真')
    sub = p.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Download upstream LAFAN1 G1 40 CSV clips and convert to standard 50 Hz NPZ')
    prepare.add_argument('--workers', type=positive, default=2, help='Concurrent downloads only; conversion is memory-bounded')
    motion = sub.add_parser('import-motion', help='Import a GMR 29-DoF G1 CSV with CPU kinematics')
    motion.add_argument('--csv', type=Path, required=True, help='Headerless xyz metres, quaternion xyzw, 29 joint radians')
    order = motion.add_mutually_exclusive_group(required=True)
    order.add_argument('--source-xml', type=Path, help='GMR assets/unitree_g1/g1_mocap_29dof.xml used for retargeting')
    order.add_argument('--g1-sdk-order', action='store_true', help='Input explicitly declares the standard 29-DoF G1 SDK joint order (e.g. NJU custom_dance)')
    motion.add_argument('--input-fps', type=float, required=True, help='Actual CSV frame rate, not the source video rate')
    motion.add_argument('--speed', type=float, default=1., help='Playback speed; 0.7 lengthens duration by 1/0.7')
    motion.add_argument('--output', type=Path, required=True, help='New output .npz; existing files are preserved')
    for command in ('train', 'resume', 'preflight'):
        child = sub.add_parser(command, parents=[train_parser()], add_help=False)
        if command == 'resume':
            child.add_argument('--checkpoint', type=Path, help='Alias for --resume; target TOTAL env steps required')
    diagnose = sub.add_parser('diagnose', parents=[inference_parser('diagnose')], add_help=False,
                              help='Measure global drift and local pose error from fixed clip starts')
    diagnose.add_argument('--output', type=Path, required=True)
    diagnose.add_argument('--seconds', type=float, default=0., help='0: whole clips; positive: truncate each clip')
    for command in ('play', 'deploy', 'evaluate'):
        child = sub.add_parser(command, parents=[inference_parser(command)], add_help=False)
        if command == 'evaluate':
            child.add_argument('--output', type=Path)
        else:
            child.add_argument('--viewer', choices=('viser', 'native'), default='viser')
            child.add_argument('--port', type=positive, default=8080)
            child.add_argument('--headless', action='store_true')
            child.add_argument('--steps', type=positive, default=500)
            child.add_argument('--reference', action=argparse.BooleanOptionalAction, default=True,
                               help='Show the synchronized blue reference ghost (default on)')
            child.add_argument('--reference-alpha', type=float, default=.35,
                               help='Reference opacity in (0, 1], default .35')
            child.add_argument('--reference-offset', type=float, nargs=3, default=(0., 0., 0.),
                               metavar=('X', 'Y', 'Z'), help='Ghost world offset in meters; default exact overlay')
    from .paths import ROOT
    showcase = sub.add_parser('showcase', help='Browser motion menu with synchronized reference; unseen motions')
    showcase.add_argument('--checkpoint', default='best')
    showcase.add_argument('--npz-dir', type=Path, default=ROOT/'data/skills')
    showcase.add_argument('--port', type=positive, default=8125)
    showcase.add_argument('--device', default='cuda:0')
    from .policy_backtrack import add_arguments
    add_arguments(sub.add_parser('blend-policy', help='Experimental CPU policy initialization; requires new evaluation'))
    from .media import add_parser
    add_parser(sub)
    from .model_download import add_parser as add_models
    add_models(sub)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == 'prepare':
            import json
            from g1_supermimic.support.datasets import prepare_lafan1
            print(json.dumps(prepare_lafan1(args.workers), ensure_ascii=False, indent=2))
        elif args.command == 'import-motion':
            import json
            from .retarget import import_motion
            print(json.dumps(import_motion(args.csv, args.source_xml, args.output,
                                           args.input_fps, args.speed), ensure_ascii=False, indent=2))
        elif args.command == 'preflight':
            from .training import preflight
            preflight(args)
        elif args.command in ('train', 'resume'):
            if args.command == 'resume':
                args.resume = args.checkpoint or args.resume
                if args.resume is None:
                    p.error('resume requires --checkpoint or --resume')
            from .training import train
            train(args)
        elif args.command == 'deploy':
            from .deployment import deploy_sim
            deploy_sim(args)
        elif args.command == 'diagnose':
            from .diagnostics import diagnose
            diagnose(args)
        elif args.command == 'showcase':
            from .showcase import play
            play(args)
        elif args.command == 'blend-policy':
            from .policy_backtrack import initialize_checkpoint
            initialize_checkpoint(args)
        elif args.command == 'record':
            from .media import record
            record(args)
        elif args.command == 'models':
            from .model_download import fetch
            fetch(args.only)
        else:
            from .inference import play, evaluate
            (evaluate if args.command == 'evaluate' else play)(args)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError) as exc:
        p.error(str(exc))
