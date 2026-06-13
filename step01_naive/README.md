# step01 — 朴素自回归推理

## 教学目标

理解自回归生成循环，直观感受 O(n²) 的性能问题。

## 自回归是什么？

模型每次只预测**下一个** token，把它加入序列，再预测下下个，如此循环：

```
输入: "Hello"
      ↓
模型 → 预测 next → token_X
      ↓
输入: "Hello token_X"
模型 → 预测 next → token_Y
      ↓
...
```

## 为什么越来越慢？

```
Step 1:  model(5 tokens)   ← 计算 5个token 的 K/V
Step 2:  model(6 tokens)   ← 计算 6个token 的 K/V  ← 重算了Step1的5个!
Step 3:  model(7 tokens)   ← 计算 7个token 的 K/V  ← 重算了Step2的6个!
...
Step n:  model((5+n) tokens)

总计算量 ∝ 5 + 6 + 7 + ... + (5+n) = O(n²)
```

## 核心代码

```python
def decode_one_step(input_ids):
    logits = model(input_ids)   # ← 全量前向，每次都重算所有 K/V！
    return torch.argmax(logits[-1])
```

## 运行

```bash
python run.py
```

## 下一步

step02 先讲采样策略（控制 token 选择方式），step03a 再解决 O(n²) 的根本问题。
