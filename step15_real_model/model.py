"""
step14: Qwen3ForCausalLM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0,
                           device=torch.device("cpu")):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs_matrix = torch.outer(positions, freqs)
    return torch.cos(freqs_matrix), torch.sin(freqs_matrix)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    seq_len, num_heads, head_dim = x.shape
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 max_seq_len=4096, rope_theta=1000000.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor, positions: Tensor,
                past_kv: Optional[Tuple[Tensor, Tensor]] = None):
        seq_len = x.size(0)
        q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q.reshape(-1, self.head_dim)).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(seq_len, self.num_kv_heads, self.head_dim)

        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=0)
            v = torch.cat([past_kv[1], v], dim=0)

        k_exp = k.repeat_interleave(self.num_groups, dim=1)
        v_exp = v.repeat_interleave(self.num_groups, dim=1)

        q_t = q.transpose(0, 1)
        k_t = k_exp.transpose(0, 1)
        v_t = v_exp.transpose(0, 1)

        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=(past_kv is None))
        out = attn_out.transpose(0, 1).reshape(seq_len, -1)
        return self.o_proj(out), (k, v)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        self.input_layernorm = RMSNorm(hidden_size, config.get("rms_norm_eps", 1e-6))
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=config.get("head_dim", hidden_size // config["num_attention_heads"]),
            max_seq_len=config.get("max_position_embeddings", 4096),
            rope_theta=config.get("rope_theta", 1000000.0),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, config.get("rms_norm_eps", 1e-6))
        self.mlp = Qwen3MLP(hidden_size, config["intermediate_size"])

    def forward(self, x, positions, past_kv=None):
        residual = x
        x = self.input_layernorm(x)
        attn_out, new_kv = self.self_attn(x, positions, past_kv)
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, new_kv


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def forward(self, input_ids, positions, past_key_values=None):
        x = self.embed_tokens(input_ids)
        new_pkv = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values else None
            x, new_kv = layer(x, positions, pkv)
            new_pkv.append(new_kv)
        x = self.norm(x)
        return self.lm_head(x), new_pkv
