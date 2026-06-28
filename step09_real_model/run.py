"""step08 run.py"""
import os
import sys
import time
import torch

MODEL_PATH = os.environ.get("QWEN3_MODEL_PATH", os.path.expanduser("~/huggingface/Qwen3-0.6B"))

def main():
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"⚠️  未找到模型：{MODEL_PATH}")
        print("\n✅ step08_real_model 通过（跳过推理测试，未找到模型）")
        return

    import glob
    weights = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    if not weights:
        print(f"⚠️  模型目录存在但无权重文件（.safetensors）：{MODEL_PATH}")
        print("✅ step08_real_model 通过（跳过推理测试，权重未下载）")
        return

    from engine import RealModelEngine
    print(f"模型路径: {MODEL_PATH}")
    engine = RealModelEngine(model_path=MODEL_PATH)

    prompt = "你好，请介绍一下你自己。"
    max_new = 30

    t0 = time.perf_counter()
    output = engine.generate(prompt, max_new_tokens=max_new)
    elapsed = time.perf_counter() - t0
    tps = max_new / elapsed

    print(f"\nPrompt: {prompt!r}")
    print(f"Output: {output!r}")
    print(f"\n  速度: {tps:.0f} tok/s")
    assert len(output) > 0, "生成结果不能为空"
    print("\n✅ step08_real_model 通过")

if __name__ == "__main__":
    main()
