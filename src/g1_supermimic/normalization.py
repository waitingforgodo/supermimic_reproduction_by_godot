"""显式更新的特征统计；forward 保留输入梯度，不修改训练或推理状态。"""
import torch
from torch import nn


class RunningNorm(nn.Module):
    """用真实数据累计总体矩，达到样本上限后冻结；统计随 checkpoint 保存。"""
    def __init__(self, dim, eps=1.e-2, max_count=200_000_000):
        super().__init__()
        if dim < 1 or eps <= 0 or max_count < 1:
            raise ValueError('RunningNorm requires positive dimensions, epsilon and sample limit')
        self.dim, self.eps, self.max_count = dim, eps, max_count
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('var', torch.ones(dim, dtype=torch.float64))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))
        self._frozen = False

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # One synchronization on load, rather than one per collection step after
        # the saved normalizer has already reached its fixed sample budget.
        self._frozen = int(self.count) >= self.max_count

    @torch.no_grad()
    def update(self, samples):
        """每个真实观测/转移仅计一次，不从重复采样的 replay minibatch 更新。"""
        if not self.training or self._frozen:
            return
        old_count = self.count.item()
        if old_count >= self.max_count:
            self._frozen = True
            return
        if samples.shape[-1] != self.dim:
            raise ValueError(f'RunningNorm expected {self.dim} features')
        samples = samples.detach().reshape(-1, self.dim).double()
        if not len(samples):
            return
        if not torch.isfinite(samples).all():
            raise FloatingPointError('Nonfinite RunningNorm input; statistics unchanged')
        variance, mean = torch.var_mean(samples, dim=0, unbiased=False)
        count = len(samples)
        total = self.count.double() + count
        delta = mean - self.mean
        new_mean = self.mean + delta * (count / total)
        new_var = (self.var * self.count + variance * count
                   + delta.square() * self.count * (count / total)) / total
        if not torch.isfinite(new_mean).all() or not torch.isfinite(new_var).all():
            raise FloatingPointError('Nonfinite RunningNorm moments; statistics unchanged')
        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(0))
        self.std.copy_(self.var.sqrt())
        self.count.add_(count)
        self._frozen = old_count + count >= self.max_count

    def forward(self, inputs):
        """固定统计下的可微标准化，eval/train 模式均不隐式更新。"""
        return (inputs - self.mean.to(inputs.dtype)) / (self.std.to(inputs.dtype) + self.eps)

    def denormalize(self, outputs):
        """把网络输出恢复到原量纲，同时保留 BPTT 梯度。"""
        return outputs * (self.std.to(outputs.dtype) + self.eps) + self.mean.to(outputs.dtype)
