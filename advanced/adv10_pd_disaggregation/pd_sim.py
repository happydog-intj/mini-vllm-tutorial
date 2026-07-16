"""
adv10_pd_disaggregation/pd_sim.py

用纯 Python 模拟 Prefill/Decode 分离部署（PD Disaggregation）。

教学版说明：
  - 不依赖真实 GPU；所有"耗时"均为数学模拟值（秒），不会实际 sleep。
  - transfer_kv 保留 time.sleep 仅演示 KV 迁移延迟概念；
    在 colocated / disaggregated 的吞吐对比中，kv_latency 以常数加入时间线。
  - 真实部署见第 5 章（DeepSeek / vLLM / Mooncake / DistServe）。
"""

import time


# ---------------------------------------------------------------------------
# 引擎抽象
# ---------------------------------------------------------------------------

class PrefillEngine:
    """
    Prefill 专用引擎（算力密集型节点）。

    prefill 的计算量 ∝ prompt_len²（注意力矩阵 n×n），
    因此耗时与 prompt 长度的平方成正比，与节点算力成反比。
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed       # 相对算力倍数，可独立配比
        self.busy = 0.0          # 累计模拟忙碌时间（秒）

    def prefill(self, prompt_len: int) -> tuple:
        """
        对长度为 prompt_len 的 prompt 做 Prefill。

        Returns
        -------
        (kv_state, t) : KV 状态字典 + 本次模拟耗时（秒）
        """
        t = (prompt_len ** 2) / (self.speed * 1e6)
        self.busy += t
        return {'kv_size': prompt_len}, t


class DecodeEngine:
    """
    Decode 专用引擎（存储带宽密集型节点）。

    每 decode step 需要读取全量 KV Cache，
    耗时 ∝ kv_size（内存带宽瓶颈），与 prompt_len 线性相关。
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.busy = 0.0

    def decode(self, kv_state: dict, steps: int) -> float:
        """
        对已有 kv_state 执行 steps 步 Decode。

        Returns
        -------
        t : 本次模拟耗时（秒）
        """
        t = kv_state['kv_size'] * steps / (self.speed * 1e6)
        self.busy += t
        return t


# ---------------------------------------------------------------------------
# KV 迁移
# ---------------------------------------------------------------------------

def transfer_kv(kv_state: dict, latency: float = 0.001) -> float:
    """
    模拟 KV Cache 从 Prefill 节点迁移到 Decode 节点。

    真实场景中通过高速网络（NVLink / RDMA）传输；
    这里用 time.sleep 演示迁移延迟概念。

    Returns
    -------
    latency : 迁移延迟（秒）
    """
    time.sleep(latency)
    return latency


# ---------------------------------------------------------------------------
# 合并部署
# ---------------------------------------------------------------------------

def colocated(
    reqs: list,
    p_speed: float = 1.0,
    d_speed: float = 1.0,
) -> tuple:
    """
    合并部署（Colocated）：同一引擎既做 Prefill 又做 Decode，严格串行。

    问题：
      - Prefill（算力密集）与 Decode（存储密集）抢同一 GPU，互相阻塞。
      - p_speed 与 d_speed 耦合（无法独立配比）；两者取同值模拟共享资源。
      - 总耗时 = Σ (t_prefill_i + t_decode_i)，无法流水线重叠。

    Parameters
    ----------
    reqs       : [(prompt_len, steps), ...]
    p_speed    : 引擎算力倍数（P/D 共用）
    d_speed    : 引擎算力倍数（P/D 共用，与 p_speed 应一致）

    Returns
    -------
    (wall_time, p_busy, d_busy) : 模拟总耗时 + 各阶段累计忙碌时间
    """
    pe = PrefillEngine(p_speed)
    de = DecodeEngine(d_speed)
    total = 0.0
    for prompt_len, steps in reqs:
        kv, t_p = pe.prefill(prompt_len)
        t_d = de.decode(kv, steps)
        total += t_p + t_d
    return total, pe.busy, de.busy


# ---------------------------------------------------------------------------
# 分离部署（流水线模拟）
# ---------------------------------------------------------------------------

def disaggregated(
    reqs: list,
    p_speed: float = 1.0,
    d_speed: float = 1.0,
    kv_latency: float = 0.001,
) -> tuple:
    """
    分离部署（Disaggregated）：P 引擎与 D 引擎独立节点，流水线重叠。

    核心收益：
      1. 流水线并行：D 解码请求 i 时，P 可已开始 Prefill 请求 i+1。
         在 prefill 远重于 decode 的长 prompt 场景，decode 耗时几乎完全
         被 prefill 时间隐藏，整体吞吐提升显著。
      2. 独立配比：可为 P 节点配更多算力（p_speed 上调），为 D 节点
         配更多内存带宽节点（d_speed 上调），各自优化，消除资源竞争。

    流水线时间线模拟（数学追踪，无实际 sleep）：

      p_time ← P 引擎的"模拟时钟"（下一个 Prefill 的开始时刻）
      d_time ← D 引擎的"模拟时钟"（下一个 Decode 的开始时刻）

      对每个请求 i：
        p_done    = p_time + t_prefill_i        # P 完成时刻
        kv_arrive = p_done  + kv_latency        # KV 到达 D 节点的时刻
        d_start   = max(d_time, kv_arrive)      # D 最早可开始时刻
        d_time    = d_start + t_decode_i        # D 完成时刻
        p_time    = p_done                      # P 立即处理下一请求

      wall_time = max(p_time, d_time)           # 所有任务最终结束时刻

    Parameters
    ----------
    reqs        : [(prompt_len, steps), ...]
    p_speed     : P 引擎算力倍数（可独立调高以应对长 prompt）
    d_speed     : D 引擎算力倍数（可独立调高以应对多步解码）
    kv_latency  : KV 迁移延迟（秒，常数近似）

    Returns
    -------
    (wall_time, p_busy, d_busy) : 流水线 makespan + 各引擎累计忙碌时间
    """
    pe = PrefillEngine(p_speed)
    de = DecodeEngine(d_speed)

    p_time = 0.0   # P 引擎模拟时钟
    d_time = 0.0   # D 引擎模拟时钟

    for prompt_len, steps in reqs:
        kv, t_p = pe.prefill(prompt_len)
        p_done = p_time + t_p
        kv_arrive = p_done + kv_latency          # KV 传输完成时刻

        d_start = max(d_time, kv_arrive)          # D 等 KV 或等自身空闲
        t_d = de.decode(kv, steps)
        d_time = d_start + t_d

        p_time = p_done                           # P 无需等 D，立即继续

    wall_time = max(p_time, d_time)
    return wall_time, pe.busy, de.busy
