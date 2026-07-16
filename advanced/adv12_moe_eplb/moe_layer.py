"""
adv12: Mixture of Experts (MoE) + Expert-Parallel Load Balancing (EPLB)

教学要点:
  - MoE 把单一 FFN 替换为多个专家 FFN,每个 token 只激活 top-k 个
  - 路由器 (gate) 输出各专家的 softmax 概率,选出最高 k 个
  - 天然存在负载不均:热门专家过载,冷门专家空闲
  - EPLB 通过贪心装箱把专家重新映射到设备,使各设备负载更均衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    Mixture of Experts 层。

    结构:
      gate  : Linear(d_model → num_experts),输出各专家路由概率
      experts: num_experts 个独立 Linear(d_model → d_model)

    前向流程:
      1. gate(x) → softmax → [seq, num_experts] 概率分布
      2. topk 选出每个 token 路由到哪 k 个专家
      3. topk_val 归一化作为加权系数
      4. 对每个专家只处理路由到它的 token(mask),加权求和写回 out
      5. 返回 out 和每个专家被选中次数 load
    """

    def __init__(self, d_model: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [seq, d_model]  — token 的隐状态

        Returns:
            out:  [seq, d_model]  — 专家加权输出
            load: [num_experts]   — 每个专家处理的 token 数量
        """
        # 路由打分: [seq, num_experts]
        scores = F.softmax(self.gate(x), dim=-1)

        # 选 top-k 专家: topk_val/topk_idx 均为 [seq, top_k]
        topk_val, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        # 归一化 top-k 权重(使权重之和 = 1)
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)

        for i in range(self.top_k):          # 第 i 个 top 选择
            for e in range(self.num_experts):  # 专家 e
                mask = (topk_idx[:, i] == e)   # 被路由到专家 e 的 token
                if mask.any():
                    # 用专家 e 处理这批 token,并按权重累加
                    out[mask] += topk_val[mask, i : i + 1] * self.experts[e](x[mask])

        # 统计每个专家被路由到的总次数
        load = torch.bincount(topk_idx.view(-1), minlength=self.num_experts)
        return out, load


def expert_imbalance(load: torch.Tensor) -> float:
    """
    负载不均衡度:最大/最小专家负载比。

    均衡时 = 1.0;值越大说明负载越倾斜。
    分母加 1e-6 防止除零(某专家 load=0 的极端情况)。
    """
    return (load.max().float() / (load.min().float() + 1e-6)).item()


def eplb_rebalance(load: torch.Tensor, num_devices: int):
    """
    EPLB 专家负载均衡 (Expert Parallel Load Balancing)。

    教学版:贪心装箱算法(Longest Processing Time First, LPT)。
    步骤:
      1. 按负载从大到小排序专家
      2. 每次把负载最大的专家分配给当前负载最小的设备
    效果:使各设备总负载尽量接近均等。

    Args:
        load:        [num_experts] 每个专家的 token 数量
        num_devices: 目标设备数

    Returns:
        assignment:  dict {expert_id -> device_id}
        device_load: list[int],每台设备的总负载
    """
    experts_sorted = torch.argsort(load, descending=True)
    device_load = [0] * num_devices
    assignment = {}

    for e in experts_sorted.tolist():
        # 选当前负载最小的设备
        d = device_load.index(min(device_load))
        assignment[e] = d
        device_load[d] += load[e].item()

    return assignment, device_load
