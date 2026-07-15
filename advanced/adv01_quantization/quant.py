"""
adv01: 量化核心逻辑

包含：
  - quantize_weight   : 对称 per-tensor 量化（INT8 / INT4）
  - dequantize_weight : 反量化回 float
  - QuantizedLinear   : W4A16 / W8A16 线性层（权重低比特存，激活 FP16，matmul 前反量化）
  - quantize_model    : 递归替换模型中所有 nn.Linear
"""

import torch
import torch.nn as nn


def quantize_weight(w: torch.Tensor, bits: int = 8):
    """
    对称 per-tensor 量化。返回 (qweight_int8, scale)。

    bits=8 → INT8，qmax=127
    bits=4 → 用 INT8 容器存 INT4，qmax=7

    公式：
      scale = max|w| / qmax
      q     = round(w / scale).clamp(-qmax, qmax)
    """
    qmax = (1 << (bits - 1)) - 1          # 7 (int4) 或 127 (int8)
    scale = w.abs().max() / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)  # int8 容器
    return q, scale


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor, bits: int = 8):
    """
    反量化：将整型权重还原为浮点。

    公式：w_approx = q * scale
    bits 参数保留以便将来做 per-channel 区分，当前实现不依赖它。
    """
    return q.to(torch.float32) * scale


class QuantizedLinear(nn.Module):
    """
    W4A16 / W8A16 量化线性层。

    继承 nn.Module 以便 setattr 可将其挂载为子模块。

    - 权重：量化后以 INT8 tensor 存储（INT4 用 INT8 容器）
    - 激活：保持输入的原始精度（FP16 / BF16 / FP32）
    - matmul 前反量化权重到激活精度，计算结果精度与输入一致

    教学版简化：
      - per-tensor 量化（一个 scale 对整层权重）
      - 真实框架用 per-channel / groupwise，每行或每组单独 scale
    """

    def __init__(self, linear: nn.Linear, bits: int = 8):
        super().__init__()
        self.bits = bits
        # 用 register_buffer 存储整型权重和 scale，随模型迁移设备
        q, scale = quantize_weight(linear.weight.data, bits)
        self.register_buffer('q', q)
        self.register_buffer('scale', scale)
        self.bias = linear.bias  # bias 保持 FP 不量化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化权重到激活精度
        w = dequantize_weight(self.q, self.scale, self.bits).to(x.dtype)
        out = x @ w.t()
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """
    递归地把模型的所有 nn.Linear 替换为 QuantizedLinear，原地操作。

    注意：named_children + setattr 只能替换直接子模块属性；
    对 TinyTransformer（单层，顶层属性）已足够。
    真实框架（如 bitsandbytes）会处理 ModuleList / Sequential 等容器。
    """
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.Linear):
            setattr(model, name, QuantizedLinear(mod, bits))
        else:
            quantize_model(mod, bits)   # 递归子模块
    return model
