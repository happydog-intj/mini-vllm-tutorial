"""
run.py — adv05 Radix Attention + CoW 教学验证脚本

场景:
  请求 A: ["sys", "prompt", "A"]
  请求 B: ["sys", "prompt", "B"]

验证断言:
  ① match_prefix(["sys","prompt","A"]) 命中长度 = 3
  ② 共享前缀节点 ["sys","prompt"] 的 ref = 2
  ③ 分叉后两个后缀节点 ("A" / "B") 各自独立，互不影响
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from radix_tree import RadixTree
from engine import RadixPrefixCacheEngine


def main() -> None:
    print("=" * 60)
    print("adv05_radix_prefix_cache — Radix Attention + CoW 验证")
    print("=" * 60)

    engine = RadixPrefixCacheEngine()

    # ----------------------------------------------------------------
    # 请求 A: ["sys", "prompt", "A"]
    # ----------------------------------------------------------------
    tokens_a = ["sys", "prompt", "A"]
    stat_a = engine.process(tokens_a)
    print(f"\n请求 A {tokens_a}:")
    print(f"  命中长度={stat_a['hit_len']}  miss 长度={stat_a['miss_len']}")
    print(f"  KV={stat_a['kv']}")

    # ----------------------------------------------------------------
    # 请求 B: ["sys", "prompt", "B"]
    # 此时树中已有 A，insert 会在 idx=2 处分叉 (CoW)
    # ----------------------------------------------------------------
    tokens_b = ["sys", "prompt", "B"]
    stat_b = engine.process(tokens_b)
    print(f"\n请求 B {tokens_b}:")
    print(f"  命中长度={stat_b['hit_len']}  miss 长度={stat_b['miss_len']}")
    print(f"  KV={stat_b['kv']}")

    # ================================================================
    # 断言 ① — match_prefix(["sys","prompt","A"]) 命中长度 = 3
    # ================================================================
    tree = engine.tree
    hit_len_a, hit_kv_a = tree.match_prefix(["sys", "prompt", "A"])
    print(f"\n[断言①] match_prefix(['sys','prompt','A']) → hit_len={hit_len_a}")
    assert hit_len_a == 3, (
        f"预期 hit_len=3，实际 hit_len={hit_len_a}"
    )
    assert hit_kv_a is not None, "预期命中的 KV 不为 None"
    print("  ✓ 命中长度正确 (3)")

    # ================================================================
    # 断言 ② — 共享前缀节点 ["sys","prompt"] 的 ref = 2
    # ================================================================
    prefix_node = engine.find_node_for_prefix(["sys", "prompt"])
    assert prefix_node is not None, (
        "找不到共享前缀节点 ['sys','prompt']，RadixTree 结构异常"
    )
    print(f"\n[断言②] 共享前缀节点 ref={prefix_node.ref}")
    assert prefix_node.ref == 2, (
        f"预期共享前缀节点 ref=2（两请求共享），实际 ref={prefix_node.ref}"
    )
    print("  ✓ 共享前缀节点 ref=2（两请求均经过该节点）")

    # ================================================================
    # 断言 ③ — 分叉后各自独立：两个后缀子节点均存在，互不影响
    # ================================================================
    children_keys = set(prefix_node.children.keys())
    print(f"\n[断言③] 分叉子节点 keys={children_keys}")

    assert "A" in children_keys, f"找不到请求 A 的后缀节点 'A'，children={children_keys}"
    assert "B" in children_keys, f"找不到请求 B 的后缀节点 'B'，children={children_keys}"

    node_a = prefix_node.children["A"]
    node_b = prefix_node.children["B"]

    # 两节点的 value 不同 → 各自持有独立 KV
    assert node_a.value is not None, "后缀节点 A 的 value 为 None"
    assert node_b.value is not None, "后缀节点 B 的 value 为 None"
    assert node_a.value is not node_b.value, (
        "后缀节点 A 和 B 共享同一 KV 对象，分叉未成功"
    )

    # 两节点的 KV 内容不同（token 序列不同导致末 token 不同）
    assert node_a.value.get(2) == "A", (
        f"后缀节点 A 的 KV[2] 预期 'A'，实际 {node_a.value.get(2)}"
    )
    assert node_b.value.get(2) == "B", (
        f"后缀节点 B 的 KV[2] 预期 'B'，实际 {node_b.value.get(2)}"
    )

    print(f"  ✓ 后缀节点 'A' 独立: key={node_a.key}, KV末token={node_a.value.get(2)!r}")
    print(f"  ✓ 后缀节点 'B' 独立: key={node_b.key}, KV末token={node_b.value.get(2)!r}")

    # ================================================================
    # 附加: 验证 ["sys","prompt","B"] 同样可以被完整命中
    # ================================================================
    hit_len_b, hit_kv_b = tree.match_prefix(["sys", "prompt", "B"])
    assert hit_len_b == 3, (
        f"预期 match_prefix(['sys','prompt','B']) hit_len=3，实际 {hit_len_b}"
    )
    print(f"\n[附加]  match_prefix(['sys','prompt','B']) → hit_len={hit_len_b} ✓")

    # ================================================================
    # 树结构概览
    # ================================================================
    print("\n树结构概览:")
    print("  root")
    root = tree.root
    for k1, n1 in root.children.items():
        print(f"  └─ key={n1.key!r}  ref={n1.ref}  value={'<kv>' if n1.value else 'None'}")
        for k2, n2 in n1.children.items():
            print(f"     └─ key={n2.key!r}  ref={n2.ref}  value={'<kv>' if n2.value else 'None'}")

    print("\n✅ adv05_radix_prefix_cache 通过")


if __name__ == "__main__":
    main()
