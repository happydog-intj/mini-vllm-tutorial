"""
adv14: Multi-LoRA 动态切换

lora.py — 核心数据结构
  - LoRALinear: 将一个 nn.Linear 包裹,附加可切换的低秩适配器(A/B 矩阵)
  - MultiLoRAEngine: 按请求 idx 动态选取 adapter

LoRA 低秩分解原理:
  原始全量微调需要更新整个权重矩阵 W (d_out × d_in)。
  LoRA 用两个小矩阵 A (d_in → r) 和 B (r → d_out) 近似增量:
      W' = W0 + BA      (r << min(d_in, d_out))
  推理时: y = W0·x + B(A(x)) = base(x) + lora_delta(x)
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    包裹一个 nn.Linear,加可切换的低秩适配器 A/B。

    参数:
        base:         原始 nn.Linear(冻结,不参与梯度更新)
        r:            LoRA 秩,越小参数越少,通常 4~64
        num_adapters: 预载的 adapter 数量(每个对应一个任务/微调版本)

    前向计算:
        y = base(x) + B(A(x))
        A: [in_features → r]     降维投影
        B: [r → out_features]    升维投影
        增量 BA 与 base 输出形状一致,可直接相加
    """

    def __init__(self, base: nn.Linear, r: int = 4, num_adapters: int = 2):
        super().__init__()
        self.base = base
        # 冻结 base 权重:LoRA 微调只训练 A/B,base 保持不变
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        in_features = base.in_features
        out_features = base.out_features

        # 为每个 adapter 创建独立的 A、B 矩阵
        # A: in_features → r  (降维)
        # B: r → out_features (升维)
        # 形状正确保证: x[*, in] → A → [*, r] → B → [*, out]
        self.adapters = nn.ModuleDict({
            str(i): nn.ModuleList([
                nn.Linear(in_features, r, bias=False),   # A: 降维
                nn.Linear(r, out_features, bias=False),  # B: 升维
            ])
            for i in range(num_adapters)
        })
        self.active = '0'  # 当前激活的 adapter 索引(字符串键)

    def set_adapter(self, idx: int) -> None:
        """切换当前激活的 adapter。推理时按请求调用,无需重载模型。"""
        self.active = str(idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        y = base(x) + B(A(x))

        base(x): 原始线性变换,结果形状 [*, out_features]
        A(x):    降维投影,    结果形状 [*, r]
        B(A(x)): 升维投影,    结果形状 [*, out_features]
        两者相加:形状匹配,LoRA 增量叠加在 base 输出之上
        """
        out = self.base(x)                        # [*, out_features]
        A, B = self.adapters[self.active]
        lora_delta = B(A(x))                      # [*, r] → [*, out_features]
        return out + lora_delta


class MultiLoRAEngine:
    """
    按请求动态选取 adapter 的推理引擎。

    真实 vLLM Multi-LoRA 会在同一批次内并发处理多个 adapter(SGMV/BGMV kernel)。
    教学版简化为:每次 generate 先 set_adapter,再做前向。

    参数:
        model:        包含 LoRALinear 层的模型
        num_adapters: 可用 adapter 数量
    """

    def __init__(self, model: nn.Module, num_adapters: int):
        self.model = model
        self.num_adapters = num_adapters

    def generate(
        self,
        token_ids: torch.Tensor,
        adapter_idx: int,
        steps: int = 4,
    ) -> torch.Tensor:
        """
        用指定 adapter 生成输出。

        参数:
            token_ids:   输入张量
            adapter_idx: 使用第几号 adapter (0 到 num_adapters-1)
            steps:       教学占位,真实场景为自回归步数

        返回:
            模型输出张量
        """
        if adapter_idx >= self.num_adapters:
            raise ValueError(
                f"adapter_idx={adapter_idx} 超出范围 [0, {self.num_adapters})"
            )

        # 遍历所有 LoRALinear 层并切换 adapter
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.set_adapter(adapter_idx)

        with torch.no_grad():
            return self.model(token_ids)
