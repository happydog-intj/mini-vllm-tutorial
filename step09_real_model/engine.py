"""
step08: RealModelEngine
"""
import json
import os
import torch
from model import Qwen3ForCausalLM
from loader import load_weights


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RealModelEngine:
    def __init__(self, model_path: str):
        self.device = _get_device()
        print(f"  使用设备: {self.device}")

        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)

        print("  初始化模型结构...")
        self.model = Qwen3ForCausalLM(config).to(torch.bfloat16).to(self.device)

        print("  加载权重...")
        load_weights(self.model, model_path)
        self.model.eval()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("  加载完成 ✅")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.device)
        prompt_len = len(ids)
        positions = torch.arange(prompt_len, device=self.device)
        logits, past_kv = self.model(ids, positions)
        nid = torch.argmax(logits[-1]).unsqueeze(0)
        generated = [nid.item()]

        for step in range(max_new_tokens - 1):
            pos = torch.tensor([prompt_len + step], device=self.device)
            logits, past_kv = self.model(nid, pos, past_key_values=past_kv)
            nid = torch.argmax(logits[-1]).unsqueeze(0)
            generated.append(nid.item())
            if nid.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated, skip_special_tokens=True)
