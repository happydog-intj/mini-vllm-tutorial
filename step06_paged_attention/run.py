import torch
from block_manager import BlockManager

def main():
    # ── 展示 1：BlockManager 基本操作 ──
    bm = BlockManager(total_blocks=10, block_size=4)
    print(f"总 Block 数: {bm.total_blocks}，Block 大小: {bm.block_size}")
    print(f"初始空闲 Block 数: {bm.num_free_blocks}")

    block_table = bm.allocate(num_blocks=2)
    print(f"\n分配 2 个 Block: {block_table}")
    print(f"剩余空闲: {bm.num_free_blocks}")

    print("\n逻辑位置 → 物理槽位翻译:")
    for token_pos in range(8):
        physical_slot = bm.translate_slot(block_table, token_pos)
        block_idx = token_pos // bm.block_size
        slot_in = token_pos % bm.block_size
        print(f"  逻辑位置 {token_pos} → Block[{block_idx}]={block_table[block_idx]} "
              f"slot {slot_in} → 物理槽位 {physical_slot}")

    bm.free(block_table)
    print(f"\n释放后空闲: {bm.num_free_blocks}")
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

    print("\n✅ step06_paged_attention 通过")

if __name__ == "__main__":
    main()
