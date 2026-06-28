"""
step06: 采样算法 — logits → next_token

教学要点:
  - Greedy: argmax，确定性，适合精确任务
  - Temperature: 控制分布尖锐程度（T→0 趋向 greedy，T→∞ 趋向均匀）
  - Top-k: 只从 top-k 个里选，截断长尾噪声
  - Top-p(Nucleus): 动态截断——累积到 p% 概率就截止
  - Gumbel-Max: 数学上等价于 temperature sampling，batch 计算更高效
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def greedy_sample(logits: Tensor) -> Tensor:
    """
    贪心采样：直接选概率最大的 token。

    logits: [vocab_size]
    返回:   标量 tensor（token_id）
    """
    return torch.argmax(logits)


def temperature_sample(logits: Tensor, temperature: float) -> Tensor:
    """
    温度采样：用 temperature 控制分布的尖锐程度。

    temperature < 1：分布更尖锐（更确定）
    temperature = 1：原始分布
    temperature > 1：分布更平坦（更随机）
    """
    assert temperature > 0, "temperature 必须 > 0"
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()


def top_k_sample(logits: Tensor, k: int, temperature: float = 1.0) -> Tensor:
    """
    Top-k 采样：只保留概率最高的 k 个 token，其余置 -inf。

    k 越小 → 越保守；k=1 → greedy；k=vocab_size → 纯 temperature
    """
    top_k_values, _ = torch.topk(logits, k=min(k, logits.size(-1)))
    threshold = top_k_values[-1]
    filtered = logits.masked_fill(logits < threshold, float("-inf"))
    return temperature_sample(filtered, temperature)


def top_p_sample(logits: Tensor, p: float, temperature: float = 1.0) -> Tensor:
    """
    Top-p（Nucleus）采样：动态保留累积概率达到 p 的最小 token 集合。

    比 top-k 更灵活：分布尖锐时保留少数 token，平坦时保留更多。
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_remove] = float("-inf")

    filtered = torch.scatter(logits.clone(), 0, sorted_indices, sorted_logits)
    return temperature_sample(filtered, temperature)


def gumbel_max_sample(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Gumbel-Max Trick：通过添加 Gumbel 噪声实现采样，数学上等价于 temperature_sample。

    原理：若 g_i ~ Gumbel(0,1)，则 argmax(logits/T + g_i) ~ Categorical(softmax(logits/T))
    优点：不需要显式 softmax + multinomial，GPU 上更高效
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20)))
    return torch.argmax(logits / temperature + gumbel_noise)
