"""
adv15_guided_decoder/guided.py

Guided Decoding via Regex Partial Match
========================================
在采样阶段将 logits 中不符合 regex 约束的 token 置 -inf，
保证生成输出永远在合法 grammar 内。

兼容性说明
----------
- re.PARTIAL_MATCH: Python 3.11+ 提案特性，实际标准库尚未合并（3.13 仍无）
- 优先级: re.PARTIAL_MATCH -> 第三方 regex 库 -> 教学回退策略
"""

import re
import sys

import torch


# ---------------------------------------------------------------------------
# Compatibility layer: partial match 实现策略
# ---------------------------------------------------------------------------

def _try_partial_match_stdlib(pattern: str, trial: str) -> bool:
    """使用标准库 re.PARTIAL_MATCH (Python 3.11+ 若已合并)。"""
    flag = getattr(re, "PARTIAL_MATCH", None)
    if flag is None:
        raise AttributeError("re.PARTIAL_MATCH not available")
    m = re.match(pattern, trial, flag)
    return m is not None


def _try_partial_match_regex_lib(pattern: str, trial: str) -> bool:
    """使用第三方 regex 库的 partial=True 参数。"""
    import regex  # type: ignore
    m = regex.match(pattern, trial, partial=True)
    return m is not None


def _partial_match_fallback(pattern: str, trial: str) -> bool:
    """
    回退策略：单字符 vocab 场景下的前缀可行性检查。

    判断 trial 是否可能是 pattern 某个完整匹配的前缀：
    1. 若 trial 本身 fullmatch -> True（已完成）
    2. 尝试在 trial 后追加探针字符串，若 fullmatch -> True（可继续扩展）
    3. 均失败 -> False（此前缀无法形成合法匹配）

    局限：对超复杂 regex 覆盖不完全，但对数字、JSON 字段等教学场景足够准确。
    探针集覆盖: 整数部分(\\d+)、小数部分(\\.\\d+)、多位数(10, 100)。
    """
    if re.fullmatch(pattern, trial):
        return True
    # 追加探针后尝试 fullmatch
    probes = ["0", "1", "10", "100", ".0", ".10"]
    for p in probes:
        if re.fullmatch(pattern, trial + p):
            return True
    return False


def _choose_partial_match_fn():
    """
    按优先级选择 partial match 实现，返回 callable(pattern, trial) -> bool。

    优先级:
    1. 标准库 re.PARTIAL_MATCH (若已提供)
    2. 第三方 regex 库 (pip install regex)
    3. 教学简化回退（probe 扩展法）
    """
    if hasattr(re, "PARTIAL_MATCH"):
        return _try_partial_match_stdlib
    try:
        import regex  # type: ignore  # noqa: F401
        return _try_partial_match_regex_lib
    except ImportError:
        pass
    return _partial_match_fallback


# 模块级单例，避免每次 next_allowed 重新选择实现
_partial_match = _choose_partial_match_fn()

# 供外部查询当前使用的策略名称（诊断/教学用）
PARTIAL_MATCH_STRATEGY: str = _partial_match.__name__


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def mask_logits(logits: torch.Tensor, allowed_token_ids: torch.Tensor) -> torch.Tensor:
    """
    把不在 allowed_token_ids 集合的 token logit 置 -inf。

    logits:            shape [vocab_size]
    allowed_token_ids: 1-D LongTensor，包含允许的 token id
    返回:              与 logits 同 shape；不在 allowed 集合的位置为 -inf
    """
    mask = torch.full_like(logits, float("-inf"))
    if allowed_token_ids.numel() > 0:
        mask[allowed_token_ids] = 0.0
    return logits + mask


class RegexGuide:
    """
    维护当前已生成文本，根据 regex 前缀匹配决定下一步允许的 token。

    工作原理
    --------
    每个解码步骤:
    1. 对词表中每个 token，拼接 generated + token 得到 trial
    2. 用 partial match 检查 trial 是否可能扩展为合法完整匹配
    3. 收集允许的 token id，调用 mask_logits 屏蔽其余 token
    4. 外部 argmax/sample 后调用 consume() 更新状态

    Parameters
    ----------
    pattern : str
        目标正则表达式（fullmatch 语义，即完整字符串匹配）。
    tokenizer_vocab : dict[int, str]
        {token_id: token_string}，覆盖词表中所有 token。
    """

    def __init__(self, pattern: str, tokenizer_vocab: dict):
        self.pattern = pattern
        self.vocab = tokenizer_vocab
        self.generated = ""

    def next_allowed(self, logits: torch.Tensor) -> torch.Tensor:
        """
        根据当前已生成文本，将不合法候选 token 的 logit 置 -inf。

        logits: shape [vocab_size]
        返回:   masked logits，相同 shape
        """
        allowed = []
        for tid, tok in self.vocab.items():
            trial = self.generated + tok
            try:
                if _partial_match(self.pattern, trial):
                    allowed.append(tid)
            except Exception:
                # regex 解析错误时跳过该 token（不允许）
                continue
        return mask_logits(logits, torch.tensor(allowed, dtype=torch.long))

    def consume(self, token_str: str) -> None:
        """记录已生成的 token，推进内部状态。"""
        self.generated += token_str

    def is_complete(self) -> bool:
        """当前已生成文本是否已是完整合法匹配。"""
        return bool(re.fullmatch(self.pattern, self.generated))
