"""step06: NaiveEngine + 可配置采样策略"""
import torch
from torch import Tensor
from model import TinyTransformer
from sampler import greedy_sample, temperature_sample, top_k_sample, top_p_sample, gumbel_max_sample


class NaiveEngine:
    def __init__(self):
        self.model = TinyTransformer()
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_gumbel: bool = False,
    ) -> Tensor:
        input_ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(input_ids)[-1]  # [vocab_size]

            if temperature == 0:
                next_id = greedy_sample(logits)
            elif use_gumbel:
                next_id = gumbel_max_sample(logits, temperature)
            elif top_k > 0:
                next_id = top_k_sample(logits, top_k, temperature)
            elif top_p < 1.0:
                next_id = top_p_sample(logits, top_p, temperature)
            else:
                next_id = temperature_sample(logits, temperature)

            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)])
        return input_ids
