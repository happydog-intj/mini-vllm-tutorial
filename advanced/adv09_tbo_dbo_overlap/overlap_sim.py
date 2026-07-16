"""
adv09_tbo_dbo_overlap/overlap_sim.py

用 Python 线程模拟 TBO（Tensor-Batch Overlap）/ DBO（Data-Batch Overlap）。
在真实框架中，计算通信重叠由 CUDA Stream 实现；
这里用 time.sleep + ThreadPoolExecutor 在 CPU 层面演示相同的调度思路。
"""

import time
from concurrent.futures import ThreadPoolExecutor


def compute(t: float) -> str:
    """模拟 attention / FFN 计算（GPU compute stream）。"""
    time.sleep(t)
    return "compute_done"


def comm(t: float) -> str:
    """模拟 dispatch / combine 通信（AllReduce / AllGather / ReduceScatter）。"""
    time.sleep(t)
    return "comm_done"


def no_overlap(microbatches: list, ct: float = 0.05, mt: float = 0.03) -> float:
    """
    朴素串行方案：每个 microbatch 先算后通信，顺序执行。

    时间轴（n=4 microbatch）：
      [C0][M0][C1][M1][C2][M2][C3][M3]
    总耗时 ≈ n * (ct + mt)
    """
    t0 = time.time()
    for _ in microbatches:
        compute(ct)
        comm(mt)
    return time.time() - t0


def tbo_overlap(microbatches: list, ct: float = 0.05, mt: float = 0.03) -> float:
    """
    TBO（Tensor-Batch Overlap）：microbatch i 的通信 与 microbatch i+1 的计算并行。

    核心思路：
      - 同时 submit 当前 microbatch 的 compute 和上一个 microbatch 的 comm
      - 两者在不同线程（类比不同 CUDA Stream）并行执行
      - 等待本轮 compute 结束后，再进入下一轮（上一轮 comm 在后台继续）

    时间轴（n=4，ct ≥ mt 时）：
      [C0]                → 第 0 轮只有计算
           [C1]           → 第 1 轮计算
           [M0]           → 第 0 轮通信 与 C1 重叠
                [C2][M1]  → 以此类推
                     [C3][M2]
                          [M3]  → 最后一轮通信单独收尾

    总耗时 ≈ n * max(ct, mt) + min(ct, mt)
             （比朴素方案节省约 n * min(ct, mt)）
    """
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        prev_comm_future = None

        for _ in microbatches:
            # 同时启动：当前 microbatch 的计算 + 上一个 microbatch 的通信
            comp_future = ex.submit(compute, ct)
            if prev_comm_future is None:
                # 第一个 microbatch：先提交计算，通信稍后启动
                prev_comm_future = ex.submit(comm, mt)
            else:
                # 后续 microbatch：上一轮通信已在运行，这里等待它完成
                # （comm 在 comp 并行期间完成；若 mt <= ct，几乎不额外等待）
                prev_comm_future.result()
                prev_comm_future = ex.submit(comm, mt)

            # 等待本轮计算完成，才能进入下一轮（流水线依赖）
            comp_future.result()

        # 收尾：等待最后一个 microbatch 的通信
        if prev_comm_future is not None:
            prev_comm_future.result()

    return time.time() - t0
