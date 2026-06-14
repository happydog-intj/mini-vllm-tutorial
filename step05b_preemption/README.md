# step05b — Preemption 抢占

## 教学目标

理解 KV Cache 耗尽时的抢占机制，系统永远不会 OOM。

## 问题

```
时刻T+10: 8个请求同时运行，KV槽位: 21/20 → 超出！

无 Preemption: RuntimeError: KV Cache 已满 💥
有 Preemption: 驱逐最低优先级请求，继续运行 ✅
```

## 抢占流程

```
检测: sum(seq.kv_len + 1 for seq in running) > max_kv_slots
         ↓
选择 victim: running 队列最后加入的（LIFO）
         ↓
驱逐: victim.free_kv_cache()     ← 释放 KV Cache
      victim.token_ids = prompt   ← 重置到 prompt 状态
      waiting.appendleft(victim)  ← 插回队首（优先恢复）
         ↓
恢复: victim 重新排到时从头 prefill
```

## 运行

```bash
python run.py
```
