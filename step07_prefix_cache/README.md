# step07 — 前缀缓存 Prefix Caching

## 教学目标

理解内容寻址缓存，相同前缀的 KV 只计算一次，多请求共享。

## 核心洞察

```
请求1: [系统提示词(100tok)] + [用户问题A(20tok)]
请求2: [系统提示词(100tok)] + [用户问题B(15tok)]
                ↑
         完全相同！→ KV 也完全相同 → 只需计算一次！
```

## 链式 xxhash（保证前缀唯一性）

```python
# 普通 hash 无法区分相同 token 的不同位置
# 链式 hash：把前一个 Block 的 hash 纳入计算
block_0_hash = xxhash64(tokens[0:B])
block_1_hash = xxhash64(str(block_0_hash) + tokens[B:2B])
# 保证：即使 token 内容相同，前缀不同 → hash 不同
```

## 适用场景

- 系统提示词（System Prompt）：所有请求共享
- Few-shot Examples：RAG 传入的文档片段
- 模板前缀：固定格式的 prompt 前半部分

## 运行

```bash
python run.py
```
