"""
step17: Column/Row Parallel Linear 层
"""
import torch
import torch.nn as nn
from torch import Tensor

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int = 1, bias: bool = False):
        super().__init__()
        assert out_features % tp_size == 0
        self.tp_size = tp_size
        self.out_features_per_rank = out_features // tp_size
        self.weight = nn.Parameter(torch.randn(self.out_features_per_rank, in_features) * 0.02)
        self.bias_param = nn.Parameter(torch.zeros(self.out_features_per_rank)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_size: int = 1, bias: bool = False):
        super().__init__()
        assert in_features % tp_size == 0
        self.tp_size = tp_size
        self.in_features_per_rank = in_features // tp_size
        self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_rank) * 0.02)
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.tp_size > 1 and DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out
