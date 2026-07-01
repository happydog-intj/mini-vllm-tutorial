"""
benchmark_indexing.py — 对比 Python 循环 vs Advanced Indexing 的 KV 写入性能

运行：
    python benchmark_indexing.py
    python benchmark_indexing.py --device cuda   # GPU 上对比更明显
"""

import argparse
import time
import torch

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
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

# ── 测试不同 seq_len ──────────────────────────────────────────────────────────
for seq_len in [16, 64, 256, 512]:
    # 构造输入
    pool = torch.zeros(args.total_blocks, args.block_size,
                       args.num_heads, args.d_head, device=device)
    K    = torch.randn(seq_len, args.num_heads, args.d_head, device=device)

    # 构造 block_table：顺序填满足够的 block
    needed_blocks = (seq_len + args.block_size - 1) // args.block_size
    block_table = list(range(needed_blocks))

    start_pos = 0

    # ── 预计算 Advanced Indexing 所需的索引 ──
    positions       = torch.arange(start_pos, start_pos + seq_len, device=device)
    block_indices   = positions // args.block_size
    slot_indices    = positions % args.block_size
    bt              = torch.tensor(block_table, device=device)
    physical_blocks = bt[block_indices]

    def run_python_loop():
        for i in range(seq_len):
            pos           = start_pos + i
            block_idx     = pos // args.block_size
            slot_in_block = pos % args.block_size
            physical      = block_table[block_idx]
            pool[physical, slot_in_block] = K[i]

    def run_advanced_indexing():
        pool[physical_blocks, slot_indices] = K

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

    t_loop = benchmark(run_python_loop,      args.warmup, args.iters)
    t_adv  = benchmark(run_advanced_indexing, args.warmup, args.iters)
    speedup = t_loop / t_adv

    print(f'seq_len={seq_len:4d} | '
          f'Python loop: {t_loop:8.2f} μs | '
          f'Advanced Indexing: {t_adv:8.2f} μs | '
          f'speedup: {speedup:5.1f}x')

print()
print('注：预计算索引（positions/block_indices/physical_blocks）的时间未计入 Advanced Indexing，')
print('    实际推理中这部分可在 engine 层复用，不是每次 forward 都重算。')
