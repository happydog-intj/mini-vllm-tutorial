"""
adv07: Sequence Parallel 模拟器

提供三个教学函数：
  - sp_attention:     Q 按序列维切分，对完整 K/V 计算 attention（与标准 attention 数值等价）
  - all_gather:       模拟跨卡 all-gather，将各卡分片拼成完整张量
  - reduce_scatter:   模拟跨卡 reduce-scatter，规约后按卡切分

真实 SP（Megatron-LM 风格）中：
  - LayerNorm 前用 all-gather 把序列分片拼回完整激活
  - Attention/MLP 结束后用 reduce-scatter 把结果切回各卡
  - 通信隐藏在计算重叠中，每层总通信量与 TP 相同
"""

import torch


def sp_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 seq_splits: int = 2) -> torch.Tensor:
    """Sequence Parallel Attention（改进教学版）。

    切分策略：Q 沿序列维（dim=0）均分为 seq_splits 段，
    每段各自对 **完整** K、V 做 scaled dot-product attention，
    结果拼回完整序列。

    为什么与标准 attention 数值等价？
    ─────────────────────────────────
    标准 attention 的第 i 行输出：
        out[i] = softmax(q[i] @ K.T / sqrt(d)) @ V

    每行只依赖 q[i]，与其他 Q 行无关。因此把 Q 切片后独立计算，
    再沿 seq 维 cat，数值与整体计算完全一致。

    教学简化：
    ─────────
    真实 SP 中每卡只持有 Q/K/V 的 1/N 分片；AllGather(K,V) 后才能
    对全序列做 attention，或者使用 Ring Attention 避免全量 KV 通信。
    本模拟假设每卡已有完整 KV 副本（模拟 AllGather 已完成），
    重点演示"Q 按序列维切分、各卡独立计算"的概念。

    参数：
        q:           shape [seq, d]
        k:           shape [seq, d]
        v:           shape [seq, d]
        seq_splits:  模拟的"卡数"（Q 的切分份数）

    返回：
        out: shape [seq, d]，与标准 attention 数值 allclose。
    """
    seq = q.size(0)
    chunk = (seq + seq_splits - 1) // seq_splits
    outs = []
    for s in range(seq_splits):
        lo = s * chunk
        hi = min((s + 1) * chunk, seq)
        if lo >= hi:
            continue
        qc = q[lo:hi]                                       # 本卡 Q 分片
        # K、V 保持完整（模拟每卡已有完整 KV 副本）
        scores = qc @ k.transpose(-2, -1) / (qc.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        outs.append(attn @ v)
    return torch.cat(outs, dim=0)


def standard_attention(q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
    """朴素标准 attention，用于对比验证。"""
    scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


def all_gather(local_shard: torch.Tensor, world_size: int = 2,
               dim: int = 0) -> torch.Tensor:
    """模拟 AllGather 集合通信。

    真实场景：每张卡持有同一张量的 1/world_size 分片，
    AllGather 后每张卡都持有完整张量。

    教学版：单进程中用 cat([shard] * world_size) 模拟"拼回完整激活"。

    参数：
        local_shard: 本卡持有的分片
        world_size:  模拟的总卡数
        dim:         沿哪个维度拼接

    返回：
        完整张量，size 在 dim 维度上为 local_shard.size(dim) * world_size。
    """
    return torch.cat([local_shard] * world_size, dim=dim)


def reduce_scatter(full: torch.Tensor, world_size: int = 2,
                   dim: int = 0) -> torch.Tensor:
    """模拟 ReduceScatter 集合通信。

    真实场景：每张卡持有完整张量的一个副本，ReduceScatter 先在所有卡上
    对应位置求和（Reduce），再把结果按卡切分（Scatter），每张卡只保留
    属于自己的那一段。

    教学版：
      1. 模拟"各卡副本相加"：sum = full * world_size（每卡都有一份 full）
      2. 取第 0 卡的分片（前 1/world_size 段）作为本卡输出。

    参数：
        full:        完整张量（代表本卡当前持有的副本）
        world_size:  模拟的总卡数
        dim:         沿哪个维度切分

    返回：
        本卡（rank 0）获得的分片，size 在 dim 维度上为 full.size(dim) // world_size。

    注：这里 sum = full * world_size 是因为 world_size 张卡各持有相同的 full，
    对应位置加和即乘以 world_size。
    """
    assert full.size(dim) % world_size == 0, (
        f"full.size({dim})={full.size(dim)} 不能被 world_size={world_size} 整除"
    )
    # 模拟 Reduce：world_size 卡各有一份 full，求和
    reduced = full * world_size
    # 模拟 Scatter：rank 0 取前 1/world_size 段
    shard_size = full.size(dim) // world_size
    return reduced.narrow(dim, 0, shard_size)
