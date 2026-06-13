# step04 — 连续批处理 Scheduler

## 教学目标

理解 Continuous Batching，实现请求调度器。

## Sequence 状态机

```
        add()          prefill 完成          is_done
WAITING ──────→ RUNNING ─────────────────→ FINISHED
                   ↑                          |
                   └── 立即补充新请求 ←────────┘
                       (Continuous Batching 核心！)
```

## Static vs Continuous Batching

```
Static（step03b）：
  槽位0 [请求A .......................完成]
  槽位1 [请求B ........完成 | padding    ]  ← 完成后空转！

Continuous（step04）：
  槽位0 [请求A .......完成][请求E ....完成][请求G ..]
  槽位1 [请求B ...完成][请求F .......完成][........]
  → 完成即释放，立即补新请求，GPU 始终满载
```

## 运行

```bash
python run.py
```
