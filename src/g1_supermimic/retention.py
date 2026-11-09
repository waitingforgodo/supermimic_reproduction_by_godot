"""每个 run 只保留经过独立整段评测的最佳、可续训 checkpoint。"""
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from .checkpoints import load_checkpoint
from .paths import ROOT

PROTOCOL = 'deterministic whole-clip reference tracking'
POLICY = dict(version=2, mode='best_only', checkpoint='best.pt', protocol=PROTOCOL,
              ranking=['whole_clip_success_rate', 'mean_clip_completion', 'mean_full_clip_norm_return'],
              evaluation='all training clips, fixed seed, independent process; every save point and final save',
              scope='one best checkpoint per run; logs and evaluation reports retained')


def atomic_json(path, value):
    """小报告同样原子替换；中断时不留下半份 JSON。"""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def full_clip_return(row):
    """Convert legacy alive-step returns without changing the native end counter.

    The native loop runs one control step per stored frame. Its original return
    divides by steps actually survived; uncompleted frames receive zero here.
    """
    if (type(row.get('failed')) is not bool or type(row.get('frames')) is not int
            or type(row.get('steps')) is not int or row['frames'] < 2 or row['steps'] < 1
            or row['steps'] > row['frames']):
        raise ValueError('Invalid whole-clip evaluation result')
    value = float(row['norm_return'])
    if not math.isfinite(value):
        raise ValueError('Nonfinite whole-clip evaluation return')
    return value * row['steps'] / row['frames']


def add_full_clip_metrics(report):
    """Keep the original return alongside an explicit full-clip denominator."""
    rows = report['clips']
    if not rows:
        raise ValueError('Whole-clip evaluation requires at least one clip')
    for row in rows:
        row['full_clip_norm_return'] = full_clip_return(row)
    report.setdefault('summary', {})['mean_full_clip_norm_return'] = (
        sum(row['full_clip_norm_return'] for row in rows) / len(rows))
    report['reward_normalization'] = dict(
        norm_return='sum reward / (max step reward * actual survived steps); legacy upstream metric',
        full_clip_norm_return='sum reward / (max step reward * reference frames); zero reward after failure',
        native_expected_steps='frames; native reference-frame step counter unchanged')
    return report


def benchmark_score(report, expected_data):
    """先整段成功，再完成比例，最后回报；拒绝缺片段或不匹配的评测。"""
    if report.get('protocol') != PROTOCOL:
        raise ValueError('Best selection requires the deterministic whole-clip protocol')
    actual_data = [(Path(x['path']).name, x['sha256']) for x in report.get('data', [])]
    expected = [(Path(x['path']).name, x['sha256']) for x in expected_data]
    if not expected or actual_data != expected:
        raise ValueError('Best selection evaluation data/order differs from the training manifest')
    rows = report.get('clips', [])
    names = [row.get('name') for row in rows]
    if len(rows) != len(expected) or len(set(names)) != len(names) or set(names) != {Path(x[0]).stem for x in expected}:
        raise ValueError('Best selection requires exactly one result for every training clip')
    success, completion, reward = 0, 0., 0.
    for row in rows:
        value = full_clip_return(row)
        success += not row['failed']
        completion += 1. if not row['failed'] else min(row['steps'] / (row['frames'] - 1), 1.)
        reward += value
    n = len(rows)
    return success / n, completion / n, reward / n


def evaluate_checkpoint(checkpoint, output, device, clips):
    """独立进程评测，训练进程的随机数、环境和采样器状态不变。"""
    output = Path(output)
    command = [sys.executable, '-m', 'g1_supermimic', 'evaluate',
               '--checkpoint', str(checkpoint), '--device', str(device),
               '--envs', str(min(40, clips)), '--output', str(output)]
    with output.with_suffix('.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Best-checkpoint evaluation exited {result.returncode}; see {output.with_suffix(".log")}')
    return json.loads(output.read_text())


def validate_confirmations(value):
    if type(value) is not int or not 1 <= value <= 5:
        raise ValueError('checkpoint_confirmations must be an integer from 1 to 5')
    return value


class BestRetention:
    """先评测、再原子发布最佳，最后清理；失败保留旧最佳和待查候选。"""
    def __init__(self, trainer, manifest, evaluator=None):
        self.trainer = trainer
        self.folder = Path(trainer.log_dir)
        self.best = self.folder / 'best.pt'
        self.candidate = self.folder / '.candidate.pt'
        self.reports = self.folder / 'selection'
        self.reports.mkdir(exist_ok=True)
        self.data = manifest['data']
        self.evaluator = evaluator or evaluate_checkpoint
        self.confirmations = validate_confirmations(getattr(trainer.cfg, 'checkpoint_confirmations', 1))
        self.score = None
        self.last_state = None
        if self.best.exists():
            checkpoint = load_checkpoint(self.best)
            self.score = benchmark_score(checkpoint['selection']['report'], self.data)

    def _index(self, checkpoint):
        record = dict(checkpoint['selection'], checkpoint=str(self.best))
        atomic_json(self.folder / 'best.json', record)

    def _publish(self, checkpoint, report, score):
        """评测与模型放进同一个原子发布的文件，JSON 只是可重建索引。"""
        import torch
        checkpoint['selection'] = dict(policy=POLICY, score=list(score), report=report,
                                       iteration=checkpoint['iteration'], num_updates=checkpoint['num_updates'])
        with tempfile.NamedTemporaryFile(dir=self.folder, prefix='.best.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, self.best)
        finally:
            temporary.unlink(missing_ok=True)
        self.score = score
        self._index(checkpoint)

    def _prune(self):
        """只删本 run 的已知中间 checkpoint，不碰其他文件或其他 run。"""
        if self.score is None or not self.best.is_file():
            return
        removed = []
        for path in self.folder.iterdir():
            if path.name == '.candidate.pt' or re.fullmatch(r'ckpt_(?:latest|[0-9]+)\.pt', path.name):
                if path.is_file() or path.is_symlink():
                    removed.append(dict(name=path.name, bytes=path.lstat().st_size))
                    path.unlink()
        return removed

    def consider(self):
        """候选可与旧最佳并存，只有完整、有限且同数据的评测才能做删除决定。"""
        import torch
        # 释放学习器的空闲缓存，为独立评测进程留出空间；活跃参数/replay 不变。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        key = (int(self.trainer.iteration), int(self.trainer.num_updates))
        output = self.reports / f'it_{key[0]:07d}_updates_{key[1]:07d}.json'
        print(f'[INFO] Evaluating checkpoint at step {key[0]} on all {len(self.data)} clips', flush=True)
        report = self.evaluator(self.candidate, output, self.trainer.device, len(self.data))
        expected = 'ema' if getattr(getattr(self.trainer.cfg, 'ema', None), 'decay', 0) else 'online'
        if report.get('policy_source', 'online') != expected:
            raise ValueError('Candidate evaluation used the wrong policy source')
        score = benchmark_score(report, self.data)
        add_full_clip_metrics(report)
        # 保存注入评测器的报告也走同一持久化路径。
        atomic_json(output, report)
        evaluations = [(report, score, output)]
        if self.score is None or score > self.score:
            for repeat in range(2, self.confirmations + 1):
                repeated_output = output.with_name(output.stem + f'_repeat_{repeat}.json')
                print(f'[INFO] Confirming prospective best: evaluation {repeat}/{self.confirmations}', flush=True)
                repeated = self.evaluator(self.candidate, repeated_output, self.trainer.device, len(self.data))
                if repeated.get('policy_source', 'online') != expected:
                    raise ValueError('Confirmation used the wrong policy source')
                repeated_score = benchmark_score(repeated, self.data)
                add_full_clip_metrics(repeated)
                atomic_json(repeated_output, repeated)
                evaluations.append((repeated, repeated_score, repeated_output))
        report, score, output = min(evaluations, key=lambda item: item[1])
        if len(evaluations) > 1:
            report['selection_validation'] = dict(aggregation='worst lexicographic score',
                runs=len(evaluations), evaluations=[dict(path=str(path), score=list(value))
                    for _, value, path in evaluations])
            atomic_json(output, report)
        promoted = self.score is None or score > self.score
        if promoted:
            checkpoint = load_checkpoint(self.candidate)
            self._publish(checkpoint, report, score)
        removed = self._prune()
        decision = dict(iteration=key[0], num_updates=key[1], candidate_score=list(score),
                        promoted=promoted, best_score=list(self.score), removed=removed,
                        evaluation=str(output), evaluations=[str(item[2]) for item in evaluations],
                        timestamp=datetime.now().isoformat())
        with (self.folder / 'selection.jsonl').open('a') as journal:
            journal.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + '\n')
        self.last_state = key
        print(f'[INFO] Best selection: success={score[0]:.3f}, completion={score[1]:.3f}, '
              f'full_clip_return={score[2]:.4f}; {"updated" if promoted else "kept previous"} best.pt', flush=True)

    def seed_resume(self, source):
        """续训先继承源最佳；旧格式先评测，避免回退后的模型覆盖它。"""
        if source is None:
            return
        checkpoint = load_checkpoint(source)
        shutil.copyfile(source, self.candidate)
        selection = checkpoint.get('selection')
        if selection:
            score = benchmark_score(selection['report'], self.data)
            add_full_clip_metrics(selection['report'])
            self._publish(checkpoint, selection['report'], score)
            self._prune()
        else:
            self.consider()


@contextmanager
def retain_best(trainer, manifest, *, evaluator=None):
    """用于所有本地 train/resume 调用，合并同一步的 latest/编号/最终保存。"""
    if not trainer.is_main:
        yield None
        return
    manager = BestRetention(trainer, manifest, evaluator)
    original_save = trainer.save

    def save(name='ckpt_latest.pt'):
        if Path(name).name != name:
            raise ValueError('Checkpoint name must be a plain filename')
        key = (int(trainer.iteration), int(trainer.num_updates))
        if manager.last_state == key:
            return
        original_save('.candidate.pt')
        try:
            manager.consider()
        except BaseException as exc:
            atomic_json(manager.folder / 'selection_error.json',
                        dict(error=f'{type(exc).__name__}: {exc}', candidate=str(manager.candidate),
                             best_checkpoint_preserved=manager.best.is_file()))
            raise

    trainer.save = save
    try:
        manager.seed_resume(getattr(trainer.cfg, 'resume', None))
        if getattr(trainer.cfg, '_init_from', None) is not None:
            # Score the pretrained policy on this task before any updates.
            # The saved checkpoint has target data, fresh counters and optimizers.
            save()
        yield manager
    finally:
        trainer.save = original_save
