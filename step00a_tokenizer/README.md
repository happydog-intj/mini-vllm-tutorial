# step00a — Tokenizer：Token 是什么

## 教学目标

理解 BPE（Byte Pair Encoding）编码原理，词表结构，encode/decode 过程。

## 核心概念

### 为什么不按字切割？

```
"Hello" → ['H','e','l','l','o']  ← 字符级：词表小但序列太长
"Hello" → ["Hello"]              ← 词语级：序列短但词表巨大（百万+）
"Hello" → [9707]                 ← BPE：平衡！常见词→单token，罕见词拆开
```

### BPE 合并规则图解

训练过程：统计语料中相邻字节对，反复合并最高频的对：

```
初始词表（字节级，256个）:
  0x48='H'  0x65='e'  0x6C='l'  0x6F='o'  ...

第1次合并（假设 'l'+'o' 最高频）:
  'H','e','l','l','o' → 'H','e','l','lo'   (lo=token_256)

第2次合并（假设 'e'+'l' 最高频）:
  'H','e','l','lo' → 'H','el','lo'          (el=token_257)
  ...
```

### 词表结构

```
token_id 0~255   → 单字节（覆盖所有字节，任何文本都能编码）
token_id 256~    → BPE 合并产生的多字节序列
                   例：token_512 = bytes("the ") （常见英文词）
```

### 特殊 Token

| Token | 含义 | 作用 |
|-------|------|------|
| `<eos>` | End of Sequence | 模型输出这个 token 时停止生成 |
| `<pad>` | Padding | Batch 推理时补齐短序列 |
| `<bos>` | Begin of Sequence | 标记序列开始 |

## 运行

```bash
python run.py
```

## 下一步

理解了 token_id 是什么之后，step00b 会解释：这个数字怎么变成向量（Embedding）。
