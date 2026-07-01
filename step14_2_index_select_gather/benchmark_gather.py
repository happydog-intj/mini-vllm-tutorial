"""
benchmark_gather.py — 对比 Python 循环 vs Advanced Indexing 的 KV gather 性能

运行：
    python benchmark_gather.py              # CPU（循环可能更快，见下方说明）
    python benchmark_gather.py --device cuda  # GPU（Advanced Indexing 大幅领先）

【为什么 CPU 上 Python 循环不一定慢？】
  - CPU 上没有 CUDA kernel launch 的固定开销（5~20μs/次）
  - pool[block_id, :slots] 在 CPU 上是连续内存 view（几乎免费）
  - torch.cat 在 CPU 上是高效 memcpy
  - Advanced Indexing 在 CPU 上是随机内存访问，cache 不友好

【为什么 GPU 上 Advanced Indexing 大幅领先？】
  - 每次 pool[block_id, :slots] 都触发一次独立的 CUDA kernel launch
  - num_blocks 次循环 = num_blocks 次 launch，固定开销叠加
  - Advanced Indexing 只有 1 次 launch，GPU 多线程并行 gather
"""

import argparse
import time
import torch

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device',       default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--total_blocks', type=int, default=128)
parser.add_argument('--block_size',   type=int, default=16)
parser.add_argument('--num_heads',    type=int, default=4)
parser.add_argument('--d_head',       type=int, default=32)
parser.add_argument('--warmup',       type=int, default=20)
parser.add_argument('--iters',        type=int, default=200)
args = parser.parse_args()

device = args.device
if device == 'cuda' and not torch.cuda.is_available():
    print('CUDA not available, falling back to CPU')
    device = 'cpu'

print(f'device={device}  total_blocks={args.total_blocks}  block_size={args.block_size}  '
      f'num_heads={args.num_heads}  d_head={args.d_head}')
print(f'warmup={args.warmup}  iters={args.iters}')
print()

# ── Python 循环版 gather ───────────────────────────────────────────────────────
def gather_loop(pool, block_table, seq_len, block_size):
    chunks = []
    remaining = seq_len
    for block_id in block_table:
        if remaining <= 0:
            break
        slots = min(block_size, remaining)
        chunks.append(pool[block_id, :slots])
        remaining -= slots
    return torch.cat(chunks, dim=0)

# ── Advanced Indexing 版 gather ───────────────────────────────────────────────
def gather_advanced(pool, block_table_tensor, seq_len, block_size):
    positions       = torch.arange(seq_len, device=pool.device)
    block_indices   = positions // block_size
    slot_indices    = positions % block_size
    physical_blocks = block_table_tensor[block_indices]
    return pool[physical_blocks, slot_indices]

# ── 测试不同 seq_len ──────────────────────────────────────────────────────────
for seq_len in [16, 64, 256, 512]:
    pool = torch.randn(args.total_blocks, args.block_size,
                       args.num_heads, args.d_head, device=device)

    needed_blocks  = (seq_len + args.block_size - 1) // args.block_size
    block_table    = list(range(needed_blocks))
    block_table_t  = torch.tensor(block_table, device=device)

    def run_loop():
        return gather_loop(pool, block_table, seq_len, args.block_size)

    def run_advanced():
        return gather_advanced(pool, block_table_t, seq_len, args.block_size)

    def benchmark(fn, warmup, iters):
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # μs per iter

    # 验证结果一致
    out_loop = run_loop()
    out_adv  = run_advanced()
    assert torch.allclose(out_loop, out_adv), 'Results differ!'

    t_loop = benchmark(run_loop,     args.warmup, args.iters)
    t_adv  = benchmark(run_advanced, args.warmup, args.iters)
    speedup = t_loop / t_adv

    num_blocks = needed_blocks
    print(f'seq_len={seq_len:4d}  num_blocks={num_blocks:2d} | '
          f'Python loop+cat: {t_loop:8.2f} μs | '
          f'Advanced Indexing: {t_adv:8.2f} μs | '
          f'speedup: {speedup:5.1f}x')

print()
print('【CPU 说明】CPU 上 Python 循环可能更快：')
print('  - pool[block_id, :slots] 是连续内存 view，几乎免费')
print('  - torch.cat 是高效 memcpy，无 kernel launch 开销')
print('  - Advanced Indexing 在 CPU 上随机访问，cache 不友好')
print()
print('【GPU 说明】GPU 上 Advanced Indexing 大幅领先：')
print('  - 每次 pool[block_id, :slots] 触发一次 CUDA kernel launch（固定开销 5~20μs）')
print('  - num_blocks 次循环 = num_blocks 次 launch，开销叠加')
print('  - Advanced Indexing 只有 1 次 launch，GPU 多线程并行 gather')
print()
print('→ 用 --device cuda 运行查看 GPU 上的真实差距。')
