"""File integrity, JSON serialization and G1 joint conventions."""
import hashlib
import json
import math
from pathlib import Path

SDK_JOINTS = (
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
)


def sha256(path):
    """分块计算文件哈希；用于动作数据、profile、checkpoint 和导出包溯源。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """把 Path/NumPy/Tensor 等转为 JSON 值；非有限浮点数显式转成文字。

    这样不会输出 JSON 标准不允许的裸 NaN/Infinity。出现非有限指标仍应
    追查源头；此函数只保证文件可读，不把异常数值伪装成有效零值。
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, 'tolist'):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, payload):
    """创建父目录并保存 UTF-8 JSON；所有输出都由调用方限定在工作空间。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def joint_permutation(source, target):
    """返回 target 每个名字在 source 数组中的索引，要求无重复且集合相同。

    target_values[i] = source_values[permutation[i]]。不能只凭两个数组长度
    相等就认为可用；缺一个腕关节也会使整套动作错位。
    """
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError('Duplicate joint names')
    if set(source) != set(target):
        raise ValueError(f'Joint sets differ: source-only={set(source)-set(target)}, target-only={set(target)-set(source)}')
    return [source.index(name) for name in target]
