"""
step09: FlashAttention 封装
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def is_flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def standard_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    if is_flash_attn_available() and q.device.type == "cuda":
        from flash_attn import flash_attn_func
        q_fa = q.transpose(1, 2)
        k_fa = k.transpose(1, 2)
        v_fa = v.transpose(1, 2)
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal)
        return out.transpose(1, 2)
    else:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
