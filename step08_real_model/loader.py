"""
step08: 从 safetensors 文件加载 HuggingFace 权重到模型
"""
import json
from pathlib import Path
from typing import Iterator, Tuple
import torch
from torch import Tensor


def iter_safetensors(model_path: str) -> Iterator[Tuple[str, Tensor]]:
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("请安装 safetensors：pip install safetensors")

    model_path = Path(model_path)
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        for shard_file in sorted(set(weight_map.values())):
            tensors = load_file(str(model_path / shard_file))
            yield from tensors.items()
    else:
        single = model_path / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"找不到权重文件: {single}")
        tensors = load_file(str(single))
        yield from tensors.items()


def load_weights(model: torch.nn.Module, model_path: str):
    state_dict = {}
    for name, tensor in iter_safetensors(model_path):
        state_dict[name] = tensor.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  权重加载完成，{len(state_dict)} 个参数张量")
    if missing:
        print(f"  未找到: {missing[:3]}")
