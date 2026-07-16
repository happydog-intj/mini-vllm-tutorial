"""
radix_tree.py — Radix Tree (基数树) for prefix KV-cache sharing with Copy-on-Write.

接口:
  RadixNode        — 树节点，含 key(token list)、value(KV cache 引用)、children、ref
  RadixTree        — 基数树
    .insert(tokens, value)               — 插入 token 序列及对应 KV
    .match_prefix(tokens) -> (len, val)  — 返回最长命中前缀长度和 KV

Plan 原始代码的 bug 修复:
  1. _split 里先截断 node.key 再用 node.key[idx] 取分叉键 → 越界。
     修复: 在截断之前保存 split_tok = node.key[idx]。
  2. _split 里 node.children 被重复赋值（覆盖），且重复赋值使用错误的 key。
     修复: 只做一次 node.children = {split_tok: suffix_child}。
  3. ref 计数缺失: insert 遍历时没有对经过的节点累加 ref，导致共享前缀节点 ref 永远是 0。
     修复: 每次 child.ref += 1（当前请求路径经过该节点即计数）。
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


class RadixNode:
    """
    基数树节点。

    key      — 该节点代表的 token 子序列（list，与 children 中的 first_token 对齐）
    value    — 对应的 KV cache 引用；内部节点（非叶）为 None
    children — 子节点字典: first_token -> RadixNode
    ref      — 引用计数：有多少条请求路径经过或终止于此节点
    """

    def __init__(self, key: List, value: Any = None) -> None:
        self.key: List = list(key)
        self.value: Any = value
        self.children: Dict[Any, "RadixNode"] = {}
        self.ref: int = 0

    def __repr__(self) -> str:
        return (
            f"RadixNode(key={self.key!r}, value={self.value!r}, "
            f"ref={self.ref}, children_keys={list(self.children.keys())})"
        )


class RadixTree:
    """Radix (compressed trie) tree — O(prefix_len) insert 和 prefix lookup。"""

    def __init__(self) -> None:
        self.root = RadixNode([])  # root 的 key 为空，不持有任何 value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, node: RadixNode, idx: int) -> RadixNode:
        """
        在 node.key[idx] 处分裂节点（Copy-on-Write 触发点）。

        Before:
            node.key = [k0 .. k_{idx-1}, k_{idx} .. k_{n-1}]
            node.value = V (或 None)
            node.children = {...}
            node.ref = R

        After:
            node.key      = [k0 .. k_{idx-1}]        (前缀，保留在 node)
            node.value    = None                       (内部节点不持有 KV)
            node.children = {k_{idx}: suffix_child}   (仅指向后缀子节点)
            node.ref      = R  (不变；已有的请求路径仍经过该前缀)

            suffix_child.key      = [k_{idx} .. k_{n-1}]
            suffix_child.value    = V
            suffix_child.children = {...}  (原 node 的 children)
            suffix_child.ref      = R      (继承；已有路径终止于此)

        返回: suffix_child
        """
        # 关键: 必须在截断 node.key 之前保存分叉键
        split_tok = node.key[idx]

        # 创建后缀子节点，继承原节点的所有子节点和 value
        suffix_child = RadixNode(node.key[idx:], node.value)
        suffix_child.children = node.children
        suffix_child.ref = node.ref  # 已有引用路径均终止于此

        # 将原节点截断为前缀内部节点
        node.key = node.key[:idx]
        node.value = None
        node.children = {split_tok: suffix_child}  # 唯一子节点
        # node.ref 保持不变：已有路径仍经过该前缀

        return suffix_child

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, tokens: List, value: Any) -> None:
        """
        插入 token 序列及对应 KV cache 引用。

        ref 计数语义:
            每个节点的 ref = 有多少条已插入的请求路径经过或终止于该节点。
            每次 insert 遍历到 child 时 child.ref += 1。
        """
        node = self.root
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok not in node.children:
                # 无匹配子节点 — 直接创建新叶节点
                leaf = RadixNode(tokens[i:], value)
                leaf.ref = 1
                node.children[tok] = leaf
                return

            child = node.children[tok]

            # 计算 child.key 与 tokens[i:] 的公共前缀长度
            j = 0
            while (
                j < len(child.key)
                and i + j < len(tokens)
                and child.key[j] == tokens[i + j]
            ):
                j += 1

            if j < len(child.key):
                # tokens 与 child.key 在位置 j 处分叉 — CoW 触发
                # 将 child 分裂: child 保留公共前缀, suffix_child 持有后缀
                self._split(child, j)

            # 当前请求路径经过 child（或终止于 child） — 引用计数 +1
            child.ref += 1
            i += j

            if i >= len(tokens):
                # tokens 恰好在 child 处结束 — 更新该节点的 KV
                child.value = value
                return

            node = child  # 继续向下遍历

    def match_prefix(self, tokens: List) -> Tuple[int, Any]:
        """
        返回最长已缓存前缀的 (命中长度, 对应 value)。

        若无任何命中，返回 (0, None)。
        只在 child.value is not None 时才更新 hit_len/hit_val，
        因此内部分裂节点（value=None）不会被错误地当成命中。
        """
        node = self.root
        i = 0
        hit_len: int = 0
        hit_val: Any = None

        while i < len(tokens):
            tok = tokens[i]
            if tok not in node.children:
                break

            child = node.children[tok]

            # 计算公共前缀长度
            j = 0
            while (
                j < len(child.key)
                and i + j < len(tokens)
                and child.key[j] == tokens[i + j]
            ):
                j += 1

            if j < len(child.key):
                # tokens 在 child 节点内部中断 — 部分匹配，停止
                break

            i += j

            if child.value is not None:
                hit_len, hit_val = i, child.value

            node = child

        return hit_len, hit_val
