"""参考 Breezewrf 的显式归一化/相对朝向实验；固定上游源码保持不变。"""
from dataclasses import dataclass, field, replace
from functools import wraps
import os
from unittest.mock import patch

import torch
from torch import nn

from g1_supermimic.core.config import TrainConfig as UpstreamConfig
from g1_supermimic.core.policy.nets import Policy, Critic
from g1_supermimic.core.world_model.world_model import WorldModel
from g1_supermimic.core.world_model.heads import OBS_TERMS
from g1_supermimic.core.kinematics.rotations import heading_quat, quat_apply_inverse, quat_inv, quat_mul, rotvec_from_quat
from .normalization import RunningNorm
from .stabilization import PolicyStabilityCfg
from .reward_alignment import ModelAlignmentCfg
from .sampling import SamplingCfg
from .ema import PolicyEMACfg
from .performance import PerformanceCfg
from .style_amp import StyleAMPCfg

REFERENCE = dict(repository='https://github.com/Breezewrf/supermimic_mjlab',
                 commit='77183062eb92026bfbee2daf276f87cad36fd957',
                 scope='Port real-data RunningNorm, dynamics LayerNorm and relative anchor orientation; retain native state features, residual integration, losses and workload')


@dataclass
class Optimizations:
    actor_input_normalization: bool = False
    critic_input_normalization: bool = False
    world_model_normalization: bool = False
    motion_anchor_ori_b: bool = False


@dataclass
class TrainConfig(UpstreamConfig):
    checkpoint_confirmations: int = 1
    optimizations: Optimizations = field(default_factory=Optimizations)
    stability: PolicyStabilityCfg = field(default_factory=PolicyStabilityCfg)
    model_alignment: ModelAlignmentCfg = field(default_factory=ModelAlignmentCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    ema: PolicyEMACfg = field(default_factory=PolicyEMACfg)
    performance: PerformanceCfg = field(default_factory=PerformanceCfg)
    style_amp: StyleAMPCfg = field(default_factory=StyleAMPCfg)


def prepare_variant(cfg, resumed=None):
    """先固定实际输入尺寸；禁止把结构变更伪装成旧模型续训。"""
    opts = cfg.optimizations
    if any(type(value) is not bool for value in vars(opts).values()):
        raise ValueError('Optimization options must be JSON booleans')
    if any(vars(opts).values()) and int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Normalization experiments currently support single-GPU training only')
    if opts.motion_anchor_ori_b:
        # 复用上游真实环境/想象环境的同一 9 维 anchor 传输；网络仅取末 6 维旋转。
        cfg.env.anchor_obs = True
    cfg.policy.goal_dim = cfg.env.future_frames * 68 + (9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    if resumed is not None:
        previous = resumed['config'].get('optimizations', vars(Optimizations()))
        if vars(opts) != previous or cfg.env.anchor_obs != resumed['config']['env'].get('anchor_obs', False):
            raise ValueError('Normalization/orientation architecture differs from the resumed checkpoint; start a new run without --resume')


def orientation_goal(goal, orientation_only):
    """6D 参考 anchor 朝向；移除传输中的 3D 相对位置，避免额外位置观测。"""
    return torch.cat([goal[..., :-9], goal[..., -6:]], dim=-1) if orientation_only else goal


def history_frames(flat, history):
    """mjlab 的逐项历史排列 -> (..., history, 93)，保持观测项顺序。"""
    parts, offset = [], 0
    for _, width in OBS_TERMS:
        size = history * width
        parts.append(flat[..., offset:offset + size].reshape(*flat.shape[:-1], history, width))
        offset += size
    if offset != flat.shape[-1]:
        raise ValueError('Unexpected term-major observation history size')
    return torch.cat(parts, dim=-1)


def term_major(frames):
    """把 (..., history, 93) 恢复为逐观测项展开的历史。"""
    return torch.cat([part.flatten(-2) for part in frames.split([d for _, d in OBS_TERMS], dim=-1)], dim=-1)


class NormalizedPolicy(Policy):
    """在原 encoder/decoder 中使用显式统计，推理与想象共享同一特征。"""
    def __init__(self, cfg, opts):
        network_cfg = replace(cfg, goal_dim=cfg.goal_dim - (3 if opts.motion_anchor_ori_b else 0))
        super().__init__(network_cfg)
        self.orientation_only = opts.motion_anchor_ori_b
        self.obs_norm = RunningNorm(93) if opts.actor_input_normalization else nn.Identity()
        self.goal_norm = RunningNorm(network_cfg.goal_dim) if opts.actor_input_normalization else nn.Identity()

    @torch.no_grad()
    def update_normalization(self, obs, goal):
        if isinstance(self.obs_norm, RunningNorm):
            # 重叠历史不重复计数，仅使用本次最新的真实观测。
            current = history_frames(obs * self.decoder.scale, self.cfg.history)[..., -1, :]
            self.obs_norm.update(current)
            encoder = self.encoders['motion']
            self.goal_norm.update(orientation_goal(goal, self.orientation_only) * encoder.scale)

    def forward(self, obs_hist, goal, interface='motion'):
        encoder = self.encoders[interface]
        goal = self.goal_norm(orientation_goal(goal, self.orientation_only) * encoder.scale)
        z = torch.tanh(encoder.mlp(goal))
        history = history_frames(obs_hist * self.decoder.scale, self.cfg.history)
        return self.decoder.mlp(torch.cat([z, term_major(self.obs_norm(history))], dim=-1))


class NormalizedCritic(Critic):
    """对原 canonical state、goal、context、root error 联合归一化。"""
    def __init__(self, cfg, pcfg, opts):
        goal_dim = pcfg.goal_dim - (3 if opts.motion_anchor_ori_b else 0)
        super().__init__(cfg, pcfg=replace(pcfg, goal_dim=goal_dim))
        self.orientation_only = opts.motion_anchor_ori_b
        self.input_norm = RunningNorm(71 + goal_dim + 33 + 4) if opts.critic_input_normalization else nn.Identity()

    def features(self, feat, goal, xi_norm, e_root):
        return torch.cat([feat, orientation_goal(goal, self.orientation_only) * self.goal_scale, xi_norm, e_root], dim=-1)

    @torch.no_grad()
    def update_normalization(self, feat, goal, xi_norm, e_root):
        if isinstance(self.input_norm, RunningNorm):
            self.input_norm.update(self.features(feat, goal, xi_norm, e_root))

    def forward(self, feat, goal, xi_norm, e_root):
        return self.mlp(self.input_norm(self.features(feat, goal, xi_norm, e_root))).squeeze(-1)


def residual_targets(spec, state, next_state):
    """精确反解上游的速度增量/显式积分残差，不能套用另一仓库的 state_delta。"""
    p, q, joints, v, w, qd = spec.split(state)
    pn, qn, jn, vn, wn, qdn = spec.split(next_state)
    heading = heading_quat(q)
    dt = spec.dt if spec.cfg.explicit_integration else 0.
    return torch.cat([
        quat_apply_inverse(heading, vn - v), wn - w, qdn - qd,
        quat_apply_inverse(heading, pn - p - dt * vn),
        rotvec_from_quat(quat_mul(quat_inv(q), qn)) - dt * wn,
        jn - joints - dt * qdn,
    ], dim=-1) / spec.out_scale


class NormalizedWorldModel(WorldModel):
    """动力学输入/输出 RunningNorm + 隐层 LayerNorm，保留原积分与损失。"""
    def __init__(self, cfg, heads, device):
        super().__init__(cfg, heads, device)
        layers = []
        original = list(self.dynamics.mlp.net.children())
        for layer in original[:-1]:
            if isinstance(layer, nn.Linear):
                layers.extend([layer, nn.LayerNorm(layer.out_features), nn.SiLU()])
        layers.append(original[-1])  # 保留零初始化的 70 维末层。
        self.dynamics.mlp.net = nn.Sequential(*layers)
        self.dynamics.input_norm = RunningNorm(133)
        self.dynamics.output_norm = RunningNorm(70, eps=1.e-4)
        # 输入拼接/反归一化仅包裹 dynamics，所有 step/compile/diagnostic 共用。
        self.dynamics.forward = self._dynamics_forward
        self.to(device)

    def _dynamics_forward(self, feat, action, xi_norm):
        dynamics = self.dynamics
        x = dynamics.input_norm(torch.cat([feat, action, xi_norm], dim=-1))
        return dynamics.output_norm.denormalize(dynamics.mlp(x))

    @torch.no_grad()
    def update_normalization(self, state, action, xi, next_state):
        features = torch.cat([self.spec.canonical(state), action, self.spec.normalize_xi(xi)], dim=-1)
        self.dynamics.input_norm.update(features)
        self.dynamics.output_norm.update(residual_targets(self.spec, state, next_state))


def make_policy(cfg):
    """按保存的配置构造网络；旧 checkpoint 保留原网络及全部键名。"""
    opts = cfg.optimizations
    return NormalizedPolicy(cfg.policy, opts) if opts.actor_input_normalization or opts.motion_anchor_ori_b else Policy(cfg.policy)


def make_critic(cfg):
    """构造与目标维度一致的原始或实验 critic。"""
    opts = cfg.optimizations
    return NormalizedCritic(cfg.critic, cfg.policy, opts) if opts.critic_input_normalization or opts.motion_anchor_ori_b else Critic(cfg.critic, pcfg=cfg.policy)


def make_trainer(cfg):
    """仅在构造期间替换三个网络，采集/优化循环继续调用固定上游实现。"""
    from g1_supermimic.core.training import loop
    if not any(vars(cfg.optimizations).values()):
        trainer = loop.Trainer(cfg)
    else:
        with patch.object(loop, 'Policy', lambda unused: make_policy(cfg)), \
             patch.object(loop, 'Critic', lambda *a, **kw: make_critic(cfg)), \
             patch.object(loop, 'WorldModel', NormalizedWorldModel if cfg.optimizations.world_model_normalization else WorldModel):
            trainer = loop.Trainer(cfg)

    from .initialization import apply_initialization
    apply_initialization(trainer)
    from .learning_rates import bind_learning_rates
    bind_learning_rates(trainer)
    from .sampling import bind_sampling
    bind_sampling(trainer)
    from .reward_alignment import bind_reward_alignment
    bind_reward_alignment(trainer)
    from .stabilization import bind_stabilization
    bind_stabilization(trainer)
    original_act = trainer.act
    @torch.no_grad()
    @wraps(original_act)
    def act(obs):
        if isinstance(trainer.policy, NormalizedPolicy):
            trainer.policy.update_normalization(obs['policy'], obs['goal'])
        if isinstance(trainer.critic, NormalizedCritic):
            from g1_supermimic.core.world_model.wm_trainer import reference_at
            recorder = trainer.recorder
            state, xi = recorder.prev_s, recorder.prev_xi
            ref = reference_at(trainer.lib, trainer.command.frame)
            trainer.critic.update_normalization(trainer.wm.spec.canonical(state), obs['goal'],
                                               trainer.wm.spec.normalize_xi(xi),
                                               trainer.wm.heads.root_error_features(state, ref))
        return original_act(obs)
    trainer.act = act

    if isinstance(trainer.wm, NormalizedWorldModel):
        recorder = trainer.recorder
        write_rows = recorder._write_rows
        @wraps(write_rows)
        def record(ids, done_next):
            # 包含 terminal 转移；在自动 reset 覆盖 s_next 前读取物理状态。
            trainer.wm.update_normalization(recorder.prev_s[ids], recorder.action_term.executed_raw[ids],
                                            recorder.prev_xi[ids], recorder.state_io.read()[ids])
            return write_rows(ids, done_next)
        recorder._write_rows = record
    from .ema import bind_ema
    from .performance import bind_performance
    bind_ema(trainer)
    from .style_amp import bind_style_amp
    bind_style_amp(trainer)
    bind_performance(trainer)
    return trainer
