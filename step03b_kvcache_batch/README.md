# step03b — 多请求 KV Cache + Batch

## 教学目标

理解多请求并发推理时 KV Cache 的管理，以及 Static Batching 的两个缺陷。

## 为什么需要 Batch？

```
串行处理（4个请求）:
  请求A [============]
                      请求B [========]
                                      请求C [======]
  GPU 利用率: ~25%  ← 一次只跑一个

Static Batch（4个请求同步）:
  请求A [============]
  请求B [========    ]  ← padding（等A完成）
  请求C [======      ]  ← padding
  请求D [==========  ]  ← padding
  GPU 利用率: ~70%  ← 提升，但有 padding 浪费
```

## Static Batching 的两大问题

1. **短请求被迫等待**：请求B完成后还要等A，GPU 槽位空转
2. **KV Cache 预分配**：每请求预留 max_len 槽位 → ~40% 内存碎片

这两个问题分别在 step04（Continuous Batching）和 step06（PagedAttention）解决。

## 运行

```bash
python run.py
```
