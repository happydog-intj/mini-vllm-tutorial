"""
engine.py — 基于 RadixTree 的前缀缓存引擎（教学版）

工作流程:
  1. 新请求到来 → match_prefix 查询已缓存的最长前缀
  2. 命中部分的 KV 直接复用（跳过 prefill）
  3. 未命中部分做模拟 prefill（用字典模拟 KV）
  4. prefill 完成后将整段 token 序列 + KV 插入 RadixTree

Copy-on-Write (CoW):
  两个请求共享公共前缀时，RadixTree 中该前缀节点的 ref=2。
  当后续 token 分叉（各自独立后缀）时，insert 在分叉点调用 _split，
  父节点保留公共前缀，两条后缀各自成为独立子节点——物理 KV 引用不拷贝，
  仅通过节点引用区分，实现零拷贝的 CoW 语义。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from radix_tree import RadixTree


class RadixPrefixCacheEngine:
    """
    教学版前缀缓存引擎。

    用 dict 模拟 KV cache（生产系统中对应显存 Block 池）。
    每次 process() 调用代表一次新推理请求。
    """

    def __init__(self) -> None:
        self._tree = RadixTree()
        # 记录每次请求的统计信息（可选，用于 run.py 验证）
        self.stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal: simulated prefill
    # ------------------------------------------------------------------

    def _sim_prefill(self, tokens: List, start: int, prefix_kv: Any) -> Any:
        """
        模拟对 tokens[start:] 的 prefill 计算。

        在教学版里 KV 用字典表示: {token_pos: token_value}。
        若有前缀 KV，先复制已有条目，再追加新 token 的 KV。

        参数:
            tokens     — 完整 token 序列
            start      — 从哪个位置开始 prefill（前 start 个已被缓存命中）
            prefix_kv  — 命中的前缀 KV（dict 或 None）
        返回:
            完整序列的 KV dict，key = 位置索引，value = token 值（模拟激活值）
        """
        kv: Dict[int, Any] = dict(prefix_kv) if prefix_kv else {}
        for pos in range(start, len(tokens)):
            kv[pos] = tokens[pos]  # 模拟: 新计算的 K/V = token 本身
        return kv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, tokens: List) -> Dict[str, Any]:
        """
        处理一条新请求。

        返回 dict:
            hit_len    — 命中前缀长度（token 数）
            miss_len   — 需要实际 prefill 的 token 数
            kv         — 完整的 KV（模拟）
            hit_ratio  — 命中率 [0, 1]
        """
        # 1. 查询已缓存的最长前缀
        hit_len, prefix_kv = self._tree.match_prefix(tokens)

        # 2. 对未命中部分做 prefill
        kv = self._sim_prefill(tokens, hit_len, prefix_kv)

        # 3. 将完整序列存入 RadixTree（触发 CoW 分叉）
        self._tree.insert(tokens, kv)

        miss_len = len(tokens) - hit_len
        hit_ratio = hit_len / len(tokens) if tokens else 0.0

        stat = {
            "tokens": tokens,
            "hit_len": hit_len,
            "miss_len": miss_len,
            "kv": kv,
            "hit_ratio": hit_ratio,
        }
        self.stats.append(stat)
        return stat

    # ------------------------------------------------------------------
    # Inspection helpers (used by run.py for assertions)
    # ------------------------------------------------------------------

    @property
    def tree(self) -> RadixTree:
        """暴露底层 RadixTree 供测试检查节点 ref 等内部状态。"""
        return self._tree

    def find_node_for_prefix(self, prefix: List) -> Optional[Any]:
        """
        返回与 prefix 完全对应的 RadixNode（供 run.py 检查 ref）。
        若找不到则返回 None。
        """
        node = self._tree.root
        i = 0
        while i < len(prefix):
            tok = prefix[i]
            if tok not in node.children:
                return None
            child = node.children[tok]
            j = 0
            while (
                j < len(child.key)
                and i + j < len(prefix)
                and child.key[j] == prefix[i + j]
            ):
                j += 1
            i += j
            if i == len(prefix):
                return child
            if j < len(child.key):
                return None  # prefix 在节点内部中断
            node = child
        return None
