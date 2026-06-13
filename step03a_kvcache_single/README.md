# step03a — 单请求 KV Cache

## 教学目标

理解 K/V 只依赖自身 token 这一关键洞察，实现单请求 KV Cache。

## 核心洞察

```
K_i = x_i · W_K     ← 只和 token i 自身有关！
V_i = x_i · W_V     ← 只和 token i 自身有关！

与 token j (j≠i) 无关  ← 不管后面来多少新 token，K_i/V_i 永远不变
```

所以：K_i 和 V_i **一旦计算，就可以永久缓存**。

## Prefill vs Decode

```
Prefill 阶段（一次性处理整个 prompt）:
  输入: [t0, t1, t2, t3, t4]
  计算: K0,V0 | K1,V1 | K2,V2 | K3,V3 | K4,V4
  存储: past_key_values ← 缓存起来

Decode 阶段（每步只传 1 个新 token）:
  Step 1: 输入 [t5]（1个token！）
    K5 = t5 · W_K   ← 只算新 token
    K_full = [K0,K1,K2,K3,K4, K5]  ← cat(past, K5)，不重算历史

  Step 2: 输入 [t6]
    K6 = t6 · W_K
    K_full = [K0,...,K5, K6]
```

## 计算量对比

```
朴素推理：总计算量 ∝ n + (n+1) + ... = O(n²)
KV Cache：每步只算 1 个新 token 的 K/V，O(1)/步
```

## 运行

```bash
python run.py
```

## 下一步

step03b：多请求 Batch 时，如何同时服务多个用户？
