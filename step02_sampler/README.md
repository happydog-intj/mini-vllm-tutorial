# step02 — 采样算法

## 教学目标

理解 logits → next_token 的各种采样策略及其适用场景。

## Logits 是什么？

模型最后输出一个 `[vocab_size]` 的向量，每个值代表该 token 的"分数"：

```
logits = [0.1, -2.3, 8.7, 0.4, ...]
           ↓     ↓    ↓    ↓
         token0 token1 token2 token3 ...
                       ↑ 分数最高！
```

## 采样策略对比

| 策略 | 公式 | 特点 | 适用场景 |
|------|------|------|----------|
| Greedy | argmax(logits) | 确定，高效 | 代码生成、摘要 |
| Temperature | softmax(logits/T) | T小→集中，T大→发散 | 通用生成 |
| Top-k | 保留top-k后temperature | 截断长尾 | 创意写作 |
| Top-p | 累积p%的集合 | 动态候选集 | 通用，最常用 |
| Gumbel-Max | argmax(logits/T+g) | 数学等价temperature，更快 | 生产系统 |

## 运行

```bash
python run.py
```
