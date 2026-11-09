"""Validate the 50 Hz G1 reference schema."""
from pathlib import Path
import numpy as np
from .contract import sha256

REQUIRED = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w',
            'body_lin_vel_w', 'body_ang_vel_w')


def scalar_float(value, name='value'):
    """Read an NPZ scalar while accepting the common one-element encoding."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f'{name} must contain exactly one value')
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def inspect(path, joint_names=None, body_names=None, expected_fps=None):
    """不允许 pickle 地读取 NPZ，验证键、名字顺序、形状、有限值和单位四元数。

    提供期望名字/频率时必须完全对应当前模型；只有列数正确还不足以训练。
    返回可保存为 JSON 的概览，不创建仿真环境。
    """
    with np.load(path, allow_pickle=False) as data:
        for name in REQUIRED + ('fps', 'joint_names', 'body_names'):
            if name not in data:
                raise ValueError(f'{path}: missing {name}; convert CSV through prepare-motion')
        joints, bodies = data['joint_names'].tolist(), data['body_names'].tolist()
        if joint_names is not None and joints != list(joint_names):
            raise ValueError('Motion joint order differs from installed robot')
        if body_names is not None and bodies != list(body_names):
            raise ValueError('Motion body order differs from installed robot')
        fps, frames = scalar_float(data['fps'],'fps'), len(data['joint_pos'])
        if fps <= 0 or not np.isfinite(fps) or frames < 3:
            raise ValueError('Invalid motion frame rate or length')
        if expected_fps is not None and not np.isclose(fps, expected_fps, rtol=1e-6):
            raise ValueError(f'Motion fps={fps}, control fps={expected_fps}; reconvert motion')
        for name in REQUIRED:
            expected = (frames, len(joints)) if name.startswith('joint_') else (frames, len(bodies), 4 if name == 'body_quat_w' else 3)
            if data[name].shape != expected or not np.isfinite(data[name]).all():
                raise ValueError(f'Invalid shape/values: {name}, expected {expected}')
        if not np.allclose(np.linalg.norm(data['body_quat_w'], axis=-1), 1, atol=1e-4):
            raise ValueError('Body quaternions are not normalized')
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path), 'frames': frames, 'fps': fps,
            'duration_seconds': (frames-1)/fps, 'joint_names': joints, 'body_names': bodies, 'quaternion': 'wxyz',
            'linear_velocity_frame': 'world', 'angular_velocity_frame': 'world'}
