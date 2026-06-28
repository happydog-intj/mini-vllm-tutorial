"""
step01: 简化版 BPE Tokenizer

教学要点:
  - Token 不是字，是「常见字节序列」
  - BPE 通过合并高频相邻字节对来构建词表
  - 字节级 BPE 保证任意 Unicode 文本都能编码（无 <unk>）
"""

from collections import Counter
from typing import Dict, List, Tuple


class SimpleBPETokenizer:
    """
    字节级 BPE Tokenizer（教学用简化版）。

    词表结构：
      0~255   → 单字节 token（基础词表，覆盖所有字节）
      256+    → BPE 合并产生的多字节 token

    与真实 tiktoken/sentencepiece 的区别：
      - 本实现不做预分词（不按空格/标点拆分）
      - 合并规则从空训练数据产生（演示用固定规则）
      - 词表大小固定为 512（256 基础 + 256 合并）
    """

    BASE_VOCAB_SIZE = 256
    MERGE_COUNT = 256  # 演示用，真实模型通常 50000+

    def __init__(self):
        # 初始化基础词表：每个字节都是一个 token
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(self.BASE_VOCAB_SIZE)}
        # 合并规则：(pair) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # 反向映射：bytes -> token_id（用于编码加速）
        self._bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # 用少量样本训练合并规则
        self._train_merges()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _train_merges(self):
        """
        教学用：用固定样本文本训练 BPE 合并规则。
        流程：
          1. 将训练文本转为字节序列
          2. 统计相邻字节对的出现频率
          3. 找到最高频的对，合并为新 token
          4. 重复直到达到目标词表大小
        """
        train_texts = [
            "the quick brown fox jumps over the lazy dog ",
            "hello world hello world hello world ",
            "Python is great for machine learning ",
            "large language model inference engine ",
        ]
        corpus: List[List[int]] = [
            list(text.encode("utf-8")) for text in train_texts
        ]

        next_id = self.BASE_VOCAB_SIZE
        for _ in range(self.MERGE_COUNT):
            if next_id >= self.BASE_VOCAB_SIZE + self.MERGE_COUNT:
                break
            pair_counts: Counter = Counter()
            for seq in corpus:
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            best_pair = pair_counts.most_common(1)[0][0]
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[next_id] = new_bytes
            self._bytes_to_id[new_bytes] = next_id
            self.merges[best_pair] = next_id
            corpus = [self._apply_merge(seq, best_pair, next_id) for seq in corpus]
            next_id += 1

    @staticmethod
    def _apply_merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """将 seq 中所有相邻的 pair 替换为 new_id。"""
        result = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return result

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token id 列表。
        步骤：
          1. UTF-8 编码 → 字节序列（每字节一个 base token）
          2. 贪心地应用合并规则（按训练顺序）
        """
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = self._apply_merge(ids, pair, new_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        将 token id 列表解码为文本。
        直接查词表拼接字节，再 UTF-8 解码。
        """
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")
