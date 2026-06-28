import torch
from block_manager import BlockManager

def main():
    # ── 展示 1：逻辑位置 vs 物理槽位不一致 ──
    # 用 total_blocks=6，block_size=4 演示非连续物理分配
    bm = BlockManager(total_blocks=6, block_size=4)
    print("=" * 55)
    print("展示：逻辑位置 ≠ 物理槽位（非连续分配）")
    print("=" * 55)
    print(f"总 Block 数: {bm.total_blocks}，Block 大小: {bm.block_size}")
    print(f"初始空闲 Block: {[b.block_id for b in bm._free]}")

    # 请求 A 先占用 block 0、1、2
    table_a = bm.allocate(num_blocks=3)
    print(f"\n请求A 分配 3 个 Block: {table_a}  ← 占用 block 0,1,2")
    print(f"剩余空闲 Block: {[b.block_id for b in bm._free]}")

    # 请求 A 中间释放 block 1（block_id=1 归还）
    bm.free([table_a[1]])
    table_a = [table_a[0], table_a[2]]  # 保留 block 0 和 2
    print(f"\n请求A 释放 block 1，当前使用: {table_a}")
    print(f"空闲 Block: {[b.block_id for b in bm._free]}  ← block 1 回到空闲池")

    # 请求 B 分配 2 个 block：拿到 block 1 和 block 3（不连续）
    table_b = bm.allocate(num_blocks=2)
    print(f"\n请求B 分配 2 个 Block: {table_b}  ← 物理上不连续！")

    print("\n逻辑位置 → 物理槽位翻译（请求B，block_size=4）:")
    print(f"  block_table = {table_b}  (逻辑block0→物理block{table_b[0]}, 逻辑block1→物理block{table_b[1]})")
    print()
    for token_pos in range(8):
        physical_slot = bm.translate_slot(table_b, token_pos)
        block_idx = token_pos // bm.block_size
        slot_in = token_pos % bm.block_size
        print(f"  逻辑位置 {token_pos} (block[{block_idx}] slot {slot_in})"
              f"  →  物理 block {table_b[block_idx]} slot {slot_in}"
              f"  →  物理槽位 {physical_slot}"
              + ("  ← 逻辑≠物理！" if physical_slot != token_pos else ""))

    print(f"\n请求A 释放: {table_a}")
    bm.free(table_a)
    print(f"请求B 释放: {table_b}")
    bm.free(table_b)
    print(f"全部释放后空闲 Block 数: {bm.num_free_blocks}")
    assert bm.num_free_blocks == bm.total_blocks

    # ── 展示 2：显存利用率对比 ──
    print("\n" + "=" * 55)
    print("显存利用率：连续分配 vs 分页分配")
    print("=" * 55)

    # 8 个请求，实际平均生成长度约 30 tokens，但连续分配要预留 max_len=50
    max_len = 50
    avg_actual = 30
    n_requests = 8
    block_size = 16

    # 连续分配：预留 max_len 槽位/请求
    total_reserved = max_len * n_requests
    total_actual = avg_actual * n_requests
    continuous_util = total_actual / total_reserved

    # 分页分配：按实际使用量分配 Block，最多浪费 block_size-1 槽/请求
    paged_waste_per_req = (block_size - (avg_actual % block_size)) % block_size
    total_paged = (avg_actual + paged_waste_per_req) * n_requests
    paged_util = total_actual / total_paged

    print(f"  连续分配: 每请求预留 {max_len} slots | 利用率 {continuous_util*100:.0f}%")
    print(f"  分页分配: 按需分配 Block（大小={block_size}）| 利用率 {paged_util*100:.0f}%")
    assert paged_util > continuous_util
    print(f"\n  提升: {continuous_util*100:.0f}% → {paged_util*100:.0f}%  ✅")

    print("\n✅ step12_paged_attention 通过")

if __name__ == "__main__":
    main()
