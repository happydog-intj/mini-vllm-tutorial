"""
adv08: Data Parallel + DPLB (Data-Parallel Load Balancing)
纯 Python 模拟多副本部署与请求路由策略对比。
"""

import time


class Replica:
    """模拟一个推理副本（实例），持有处理队列和飞行中的 token 数。"""

    def __init__(self, name: str, speed: float = 1.0):
        self.name = name
        self.speed = speed          # 处理速率倍率（异构副本）
        self.queue: list[int] = []  # 待处理请求大小列表（token 数）
        self.in_flight: int = 0     # 当前正在处理的 token 数

    def load(self) -> int:
        """当前总负载 = 飞行中 + 队列中所有 token。"""
        return self.in_flight + sum(self.queue)

    def step(self, dt: float = 1.0) -> None:
        """推进一个时间步，按 speed 消耗 in_flight token，从队列补充。"""
        # 本步消耗量
        done = min(self.in_flight, int(self.speed * dt * 100))
        self.in_flight -= done
        # 从队列补充，直到 in_flight 达到上限
        while self.queue and self.in_flight < 100:
            self.in_flight += self.queue.pop(0)


class RoundRobinLB:
    """朴素轮询负载均衡：按顺序依次分配，不感知副本负载。"""

    def __init__(self):
        self.i = 0

    def route(self, replicas: list[Replica], req_size: int) -> Replica:
        r = replicas[self.i % len(replicas)]
        self.i += 1
        return r


class LeastLoadLB:
    """最小负载均衡（DPLB 核心）：每次路由到当前负载最低的副本。"""

    def route(self, replicas: list[Replica], req_size: int) -> Replica:
        return min(replicas, key=lambda r: r.load())


def simulate(
    replicas: list[Replica],
    lb,
    arrivals: list[tuple[int, int]],
    total_time: int = 50,
) -> list[dict[str, int]]:
    """
    模拟请求到达与副本处理过程。

    参数:
        replicas: 副本列表（每次调用前应重置状态）
        lb: 负载均衡器实例
        arrivals: [(到达时刻, 请求大小), ...]，按时刻升序
        total_time: 总模拟时间步数

    返回:
        log: 每个时间步各副本的负载快照，list[{replica_name: load}]
    """
    arrivals = list(arrivals)  # 复制，避免修改原数据
    log: list[dict[str, int]] = []

    for t in range(total_time):
        # 注入当前时刻到达的请求
        while arrivals and arrivals[0][0] == t:
            _, size = arrivals.pop(0)
            r = lb.route(replicas, size)
            r.queue.append(size)
        # 所有副本推进一步
        for r in replicas:
            r.step(1)
        # 记录快照
        log.append({r.name: r.load() for r in replicas})

    return log
