"""
step03a: NaiveEngine（对照）+ KVCacheEngine（本步核心）

KV Cache 关键洞察：
  K_i = x_i · W_K   ← 只取决于 x_i 自身，与后续 token 无关
  V_i = x_i · W_V   ← 只取决于 x_i 自身

  → K_i/V_i 一旦计算，可以永久缓存，无需重算
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch import Tensor

# 对照组：沿用 step01 的朴素引擎
import importlib.util as _ilu
_step01_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'step01_naive')
_step01_path = os.path.join(_step01_dir, 'engine.py')
sys.path.insert(0, os.path.abspath(_step01_dir))
_spec = _ilu.spec_from_file_location('step01_engine', _step01_path)
_step01_engine = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_step01_engine)
sys.path.pop(0)
# Clean up any 'model' cached by step01's exec so our local model.py is found
sys.modules.pop('model', None)
NaiveEngine = _step01_engine.NaiveEngine

# Import our model explicitly to avoid sys.path confusion
_this_dir = os.path.dirname(os.path.abspath(__file__))
_model_spec = _ilu.spec_from_file_location('step03a_model', os.path.join(_this_dir, 'model.py'))
_model_mod = _ilu.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_mod)
TinyTransformerWithKVCache = _model_mod.TinyTransformerWithKVCache


class KVCacheEngine:
    """
    带 KV Cache 的推理引擎。

    generate 的两阶段：
      1. Prefill：所有 prompt token 一次前向，计算并缓存所有 K/V
      2. Decode：每步只传 1 个新 token，K/V 只算这 1 个，历史从 cache 读取
    """

    def __init__(self):
        self.model = TinyTransformerWithKVCache()
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> Tensor:
        # ── Prefill ──
        logits, past_key_values = self.model(prompt_ids, past_key_values=None)
        next_id = self._sample(logits[-1], temperature)
        generated = [next_id]

        # ── Decode ──
        for _ in range(max_new_tokens - 1):
            # 只传 1 个 token！
            logits, past_key_values = self.model(
                next_id.unsqueeze(0),
                past_key_values=past_key_values,
            )
            next_id = self._sample(logits[-1], temperature)
            generated.append(next_id)

        return torch.cat([prompt_ids, torch.stack(generated)])

    def _sample(self, logits: Tensor, temperature: float) -> Tensor:
        if temperature == 0:
            return torch.argmax(logits)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze()
