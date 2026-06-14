# step06 — Block Manager + PagedAttention

## 教学目标

理解分页内存管理，实现 BlockManager，彻底消除 KV Cache 碎片。

## OS 分页类比

```
操作系统虚拟内存            PagedAttention KV Cache
────────────────────      ──────────────────────────
物理内存页帧 (4KB)    ←→  KV Block（block_size 个token的K/V）
虚拟地址              ←→  逻辑 token 位置 (0, 1, 2, ...)
页表 (page table)     ←→  block_table: List[int]
```

## block_table 图解

```
Sequence A: token_ids = [t0, t1, t2, t3, t4, t5]  (block_size=4)

block_table = [7, 3]   ← 占用物理 Block 7 和 Block 3

逻辑位置 0: Block[0]=7 → 物理槽位 7*4+0=28
逻辑位置 4: Block[1]=3 → 物理槽位 3*4+0=12
```

## 利用率提升原理

```
连续分配：每请求预留 max_len=50，实际用 30 → 40% 浪费
分页分配：按需分配，最多浪费 block_size-1=15 → 利用率 ~95%
```

## 运行

```bash
python run.py
```
