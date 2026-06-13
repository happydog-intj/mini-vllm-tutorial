# step05a — Chunked Prefill

## 教学目标

理解长 prompt 如何分块处理，避免阻塞已有请求的 decode。

## 问题

```
时刻 T+1：来了 1 个 200-token 的超长 prompt

无 Chunked Prefill：
  [prefill 200 token... 需要很长时间...]
  ← 已有 4 个 decode 请求被完全阻塞！

有 Chunked Prefill（chunk_size=50）：
  步骤1: [prefill 50 token][decode A/B/C/D]  ← 混合调度！
  步骤2: [prefill 50 token][decode A/B/C/D]
  步骤3: [prefill 50 token][decode A/B/C/D]
  步骤4: [prefill 50 token][decode A/B/C/D]
  → 长 prompt 被"稀释"，不影响 decode
```

## 关键参数

- `chunk_size`：每步最多处理的 prefill token 数
- `prefill_offset`：当前序列已 prefill 了多少 token

## 运行

```bash
python run.py
```
