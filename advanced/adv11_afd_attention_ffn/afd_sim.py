"""
adv11: AFD (Attention-FFN Disaggregation) 核心模拟

教学要点:
  - 一个 Transformer 层 = Attention 子模块 + FFN 子模块
  - 两个子模块计算特性不同 (Attention: 访存密集; FFN: 计算密集)
  - AFD 把 Attention 和 FFN 部署到不同设备集群
  - balanced_config 计算 A/F 配比使两端实际耗时均衡

注意: 本模块为纯 Python 模拟,用 time.sleep 代替真实 GPU 计算,
      旨在建立直觉而非衡量真实性能。
"""

import math
import time

import torch


class AttentionDevice:
    """
    模拟运行 Attention 子模块的设备(组)。

    真实场景: 对应 step04 TransformerDecoderLayer 中的
              norm1 + MultiHeadAttention 部分,
              在 AFD 框架中被分配到专属 Attention GPU 集群。
    """

    def __init__(self, n: int = 1, t: float = 0.02):
        """
        Args:
            n: 并行单元数(对应 Attention 专用设备数量)。
            t: 单设备完整计算耗时(秒)。n 个设备并行后耗时变为 t/n。
        """
        self.n = n
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 Attention 计算, 耗时 t/n 秒(多设备并行摊薄)。

        真实应做: Q/K/V 投影 + scaled dot-product attention + 输出投影。
        此处用 time.sleep 占位, 结果直接返回输入张量。
        """
        time.sleep(self.t / self.n)
        return x


class FFNDevice:
    """
    模拟运行 FFN(MLP) 子模块的设备(组)。

    真实场景: 对应 step04 TransformerDecoderLayer 中的
              norm2 + MLP(SwiGLU) 部分,
              在 AFD 框架中被分配到专属 FFN GPU 集群。
    """

    def __init__(self, n: int = 1, t: float = 0.03):
        """
        Args:
            n: 并行单元数(对应 FFN 专用设备数量)。
            t: 单设备完整计算耗时(秒)。n 个设备并行后耗时变为 t/n。
        """
        self.n = n
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 FFN 计算, 耗时 t/n 秒(多设备并行摊薄)。

        真实应做: W_gate/W_up 投影 + SiLU 激活 + W_down 投影(SwiGLU)。
        此处用 time.sleep 占位, 结果直接返回输入张量。
        """
        time.sleep(self.t / self.n)
        return x


def run_layer(
    seq_len: int,
    attn_dev: AttentionDevice,
    ffn_dev: FFNDevice,
) -> torch.Tensor:
    """
    模拟一个 Transformer 层 = Attention + FFN 顺序执行。

    数据流 (对应 step04 TransformerDecoderLayer, 残差/Norm 已省略):
      x → AttentionDevice.forward → FFNDevice.forward → 返回

    Args:
        seq_len:  序列长度 (用于创建占位张量)。
        attn_dev: Attention 设备组实例。
        ffn_dev:  FFN 设备组实例。

    Returns:
        经过一层处理的张量 (此模拟中值不变)。
    """
    x = torch.zeros(seq_len)
    x = attn_dev.forward(x)
    x = ffn_dev.forward(x)
    return x


def balanced_config(attn_time: float, ffn_time: float) -> tuple[int, int]:
    """
    计算均衡 A/F 设备配比, 使两端实际耗时接近(利用率均衡)。

    核心思路:
      若 a_units 个 Attention 设备并行, 则实际耗时 = attn_time / a_units。
      若 f_units 个 FFN      设备并行, 则实际耗时 = ffn_time  / f_units。
      均衡目标: attn_time / a_units ≈ ffn_time / f_units
      即:       a_units : f_units   = attn_time : ffn_time

    实现:
      将两个耗时缩放到整数(精度 0.1ms), 再求最大公因数约分,
      得到最小整数比, 避免浮点精度误差。

    Args:
        attn_time: Attention 单设备耗时(秒)。
        ffn_time:  FFN      单设备耗时(秒)。

    Returns:
        (a_units, f_units): 建议的 Attention/FFN 设备数量。

    示例:
        attn_time=0.02, ffn_time=0.05
        → scale: 200, 500  → gcd=100 → (2, 5)
        → 验证: 0.02/2 = 0.01s = 0.05/5  ✓ 两端耗时相等
    """
    # 缩放到整数 (精度 0.1ms = 1e-4s), 避免浮点精度问题
    scale = 10_000
    a_int = max(1, round(attn_time * scale))
    f_int = max(1, round(ffn_time * scale))
    g = math.gcd(a_int, f_int)
    return a_int // g, f_int // g
