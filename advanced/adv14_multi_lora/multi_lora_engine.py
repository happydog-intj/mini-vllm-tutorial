"""
adv14: Multi-LoRA 动态切换

multi_lora_engine.py — 批量调度演示

将 MultiLoRAEngine 的使用场景具体化:
  - 模拟多请求并发,每个请求携带不同的 adapter_idx
  - 展示如何在推理服务层按请求路由到不同 LoRA 适配器

教学版采用单线程顺序调度,真实框架(vLLM)使用 SGMV/BGMV CUDA kernel
在同一批次内同时执行多个 adapter 的矩阵乘法。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List

from lora import LoRALinear, MultiLoRAEngine


@dataclass
class Request:
    """模拟推理请求:携带输入张量和所需的 adapter 编号。"""
    request_id: str
    token_ids: torch.Tensor
    adapter_idx: int


def build_demo_model(in_features: int = 8, out_features: int = 8,
                     r: int = 2, num_adapters: int = 2) -> nn.Sequential:
    """构建含单层 LoRALinear 的演示模型。"""
    base_linear = nn.Linear(in_features, out_features, bias=False)
    lora_layer = LoRALinear(base_linear, r=r, num_adapters=num_adapters)
    return nn.Sequential(lora_layer)


def schedule_requests(
    engine: MultiLoRAEngine,
    requests: List[Request],
) -> dict:
    """
    顺序调度请求列表,返回 {request_id: output_tensor}。

    真实 vLLM Multi-LoRA 调度:
      1. 按 adapter_idx 对请求分组
      2. 同组请求组成一个 batch,用 BGMV kernel 并行计算 LoRA 增量
      3. 所有组的结果拼回原始顺序返回

    教学版:逐请求顺序处理,adapter 切换开销 O(层数)。
    """
    results = {}
    for req in requests:
        out = engine.generate(req.token_ids, req.adapter_idx)
        results[req.request_id] = out
        print(
            f"  请求 {req.request_id} | adapter={req.adapter_idx} "
            f"| out_norm={out.norm().item():.4f}"
        )
    return results


if __name__ == "__main__":
    torch.manual_seed(0)

    model = build_demo_model(in_features=8, out_features=8,
                             r=2, num_adapters=3)
    engine = MultiLoRAEngine(model, num_adapters=3)

    x = torch.randn(1, 8)
    requests = [
        Request("req-A", x.clone(), adapter_idx=0),
        Request("req-B", x.clone(), adapter_idx=1),
        Request("req-C", x.clone(), adapter_idx=2),
        Request("req-D", x.clone(), adapter_idx=0),  # 同一 adapter 复用
    ]

    print("=== Multi-LoRA 批量调度演示 ===")
    results = schedule_requests(engine, requests)

    # 验证相同 adapter 产生相同结果
    assert torch.allclose(results["req-A"], results["req-D"]), \
        "相同 adapter 应产生相同输出"
    # 验证不同 adapter 产生不同结果
    assert not torch.allclose(results["req-A"], results["req-B"]), \
        "不同 adapter 应产生不同输出"

    print("\n✅ multi_lora_engine 调度验证通过")
