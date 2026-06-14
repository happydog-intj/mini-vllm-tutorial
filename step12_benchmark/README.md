# Step 12: Benchmark — 推理性能评测

## 本节目标

实现标准化的 LLM 推理 Benchmark 工具，测量 TTFT、TPOT、吞吐量三个核心指标。

## 核心指标

### TTFT（Time To First Token）首 Token 延迟

从发送请求到收到第一个输出 token 的时间。

- 主要受 **prefill 阶段**影响（处理输入 prompt）
- 输入越长，TTFT 越高
- 典型值：100ms ~ 2000ms（取决于模型大小和输入长度）

### TPOT（Time Per Output Token）每 Token 延迟

生成每个后续 token 的平均时间。

- 主要受 **decode 阶段**影响
- 决定用户感知的"流式输出速度"
- 典型值：5ms ~ 50ms/token

### 吞吐量（Throughput）

单位时间内处理的总 token 数（output tokens/s）。

- 衡量系统整体处理能力
- 通过**连续批处理**（continuous batching）最大化
- 典型值：100 ~ 10000 tok/s（取决于硬件）

## 指标关系

```
TTFT ↑  →  用户等待第一个字变长（体验差）
TPOT ↑  →  流式输出变慢（体验差）
吞吐量 ↑  →  能同时服务更多用户（成本低）

TPOT 与吞吐量存在 trade-off：
  大 batch → 高吞吐，但 TPOT 也更高
  小 batch → 低 TPOT，但吞吐量低
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `bench.py` | Benchmarker 类，支持随机请求模拟 |
| `run.py` | 运行 benchmark 并打印统计报告 |

## 运行

```bash
python run.py
# 无模型时使用模拟数据展示报告格式
```
