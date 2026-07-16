"""
pp_sim.py — Pipeline Parallel 调度模拟器（教学版）

支持两种调度策略:
  GPipe   — 全前向完成后统一反向，显存峰值 = num_microbatches
  1F1B    — Warmup + Steady-State + Cooldown，显存峰值 = pipeline_stages - 1

注意: 本模拟器为单机串行仿真，所有 stage 顺序执行，无真实跨 GPU 通信。
     显存峰值差异须通过理论分析（compute_theoretical_peak）来体现，
     因为串行执行无法直接测量出并行运行时各 stage 的激活并发驻留情况。
"""


class Device:
    """模拟 pipeline 中的一个 GPU stage。"""

    def __init__(self, name, layers, comm_latency=0.01, fwd_time=0.02, bwd_time=0.04):
        self.name = name
        self.layers = layers
        self.comm_latency = comm_latency  # stage 间通信延迟（秒）
        self.fwd_time = fwd_time          # 单 microbatch 前向时间（秒）
        self.bwd_time = bwd_time          # 单 microbatch 反向时间（秒）


def gpipe_schedule(devices, num_microbatches):
    """GPipe: 所有 microbatch 正向跑完，再统一反向。

    调度顺序（p=4, n=4 为例）:
      F(mb0) on D0..D3 → F(mb1) on D0..D3 → ... → F(mb3) → B(mb0) → ...

    显存分析:
      每个 stage 需在整个 forward 阶段（所有 n 个 mb）结束后才开始 backward，
      因此同时驻留的激活数 = n（全部 microbatch）。

    Returns:
        total_time (float): 串行模拟总时间（秒）
        events (list): 事件列表 [(time, device_name, phase, microbatch_id), ...]
    """
    events = []
    t = 0.0
    for mb in range(num_microbatches):
        for d in devices:
            t += d.comm_latency
            events.append((t, d.name, 'F', mb))
            t += d.fwd_time
    for mb in range(num_microbatches):
        for d in reversed(devices):
            t += d.comm_latency
            events.append((t, d.name, 'B', mb))
            t += d.bwd_time
    return t, events


def onef_oneb_schedule(devices, num_microbatches):
    """1F1B (One-Forward-One-Backward): warmup + steady-state + cooldown。

    调度结构（p=4, n=4 为例）:
      Warmup  : F(mb0) F(mb1) F(mb2)          ← 填充 pipeline，送入 p-1 个 mb
      Steady  : F(mb3) B(mb0)                  ← 新 F 与最旧 B 交替，控制驻留数
      Cooldown:         B(mb1) B(mb2) B(mb3)   ← 排空 pipeline

    显存分析（stage 0 视角）:
      - Warmup 深度 = p-1（而非 GPipe 的 n），warmup 结束即开始 backward
      - 稳态下 stage 0 同时驻留的激活数 = p-1（vs GPipe 的 n）
      - 当 n >> p 时（如 n=32, p=4），节省显著：峰值从 32 降至 3

    ⚠️ 教学版局限性:
      本实现为串行顺序模拟——所有 stage 串行执行，无并行。
      因此串行测量的总时间 **不等于** 真实多 GPU 上的 1F1B 执行时间。
      显存优势须通过 compute_theoretical_peak() 进行理论分析，而非实测。
      真实 1F1B 调度见 Megatron-LM / DeepSpeed PP 实现。

    Returns:
        total_time (float): 串行模拟总时间（秒）
        events (list): 事件列表 [(time, device_name, phase, microbatch_id), ...]
    """
    events = []
    t = 0.0
    p = len(devices)
    n = num_microbatches

    ops = []  # (device, 'F'|'B', mb_id)

    # Warmup: 先送入 (p-1) 个 mb 做前向，填满 pipeline
    for mb in range(p - 1):
        for d in devices:
            ops.append((d, 'F', mb))

    # Steady state: 每引入一个新 mb 做 F，立即对最旧 pending mb 做 B
    oldest_bwd = 0
    for mb in range(p - 1, n):
        for d in devices:
            ops.append((d, 'F', mb))
        for d in reversed(devices):
            ops.append((d, 'B', oldest_bwd))
        oldest_bwd += 1

    # Cooldown: 对所有剩余 pending mb 做 B，排空 pipeline
    for mb in range(oldest_bwd, n):
        for d in reversed(devices):
            ops.append((d, 'B', mb))

    # 将操作序列转换为带时间戳的事件
    for (d, phase, mb) in ops:
        t += d.comm_latency
        events.append((t, d.name, phase, mb))
        t += d.fwd_time if phase == 'F' else d.bwd_time

    return t, events


def compute_peak_resident(events, device_name):
    """从事件列表统计指定设备的实测峰值驻留 microbatch 数。

    驻留定义: microbatch 在该设备完成 F（激活需保存）但未完成 B 的时间段。

    ⚠️ 串行模拟限制:
      由于串行执行，同一设备的所有 F 事件均早于所有 B 事件，
      两种调度的实测峰值均等于 n（无法体现 1F1B 的并行显存优势）。
      并行场景下的理论峰值请使用 compute_theoretical_peak()。
    """
    fwd_done = {}  # mb -> F 完成时间
    bwd_done = {}  # mb -> B 完成时间

    for (t, name, phase, mb) in events:
        if name == device_name:
            if phase == 'F':
                fwd_done[mb] = t
            elif phase == 'B':
                bwd_done[mb] = t

    all_times = sorted(set(list(fwd_done.values()) + list(bwd_done.values())))
    peak = 0
    for t in all_times:
        resident = sum(
            1 for mb in fwd_done
            if fwd_done[mb] <= t and (mb not in bwd_done or bwd_done[mb] > t)
        )
        peak = max(peak, resident)
    return peak


def compute_theoretical_peak(schedule_type, num_microbatches, num_stages):
    """计算调度策略的理论显存峰值（同时驻留的 microbatch 数）。

    理论推导（以 stage 0 的激活驻留为基准）:

      GPipe:
        所有 n 个 mb 的前向全部完成后才开始任何反向。
        → stage 0 峰值 = n。

      1F1B:
        stage 0 仅需完成 (p-1) 个 mb 的前向（warmup）后即交替 F/B。
        → stage 0 峰值 = p-1。

    其中 p = num_stages（pipeline 深度）。

    显存节省分析:
      - n = p 时：GPipe 峰值=n，1F1B 峰值=p-1，节省 1 个 mb 激活。
      - n = 8, p = 4 时：GPipe=8，1F1B=3，节省 62.5%。
      - n = 32, p = 4 时：GPipe=32，1F1B=3，节省 90.6%。

    Args:
        schedule_type: 'gpipe' 或 '1f1b'
        num_microbatches: microbatch 数量 (n)
        num_stages: pipeline stage 数量 (p)

    Returns:
        int: 理论峰值驻留 microbatch 数
    """
    if schedule_type == 'gpipe':
        return num_microbatches
    elif schedule_type == '1f1b':
        return num_stages - 1
    else:
        raise ValueError(f"未知调度类型: {schedule_type!r}，支持 'gpipe' 或 '1f1b'")
