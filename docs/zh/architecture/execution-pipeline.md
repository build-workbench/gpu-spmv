# 执行流水线

## 为什么要单独讲执行流水线

GPU SpMV 的难点不在于“调用一个 kernel”，而在于 **输入矩阵如何被分析、如何做 kernel 选择、如何复用执行上下文，以及如何解释结果是否可信**。

## Pipeline 分解

1. **输入阶段**：加载 CSR / ELL 数据结构，准备输入向量。
2. **分析阶段**：统计 `avg_nnz_per_row`、偏斜度和行分布模式。
3. **决策阶段**：基于统计结果选择 Scalar CSR、Vector CSR、Merge Path 或 ELL。
4. **执行阶段**：调度 GPU kernel，记录时间和带宽指标。
5. **验证阶段**：与 CPU 参考结果或既有基线做一致性检查。

## 关键判断

| 现象 | 决策 |
|:-----|:-----|
| `avg_nnz_per_row < 4` | Scalar CSR，避免 warp 级资源浪费 |
| 行长度均匀、偏斜度低 | Vector CSR，提升 warp 内协作效率 |
| 行长度高度不均 | Merge Path，优先负载均衡 |
| 行宽近似固定 | ELL kernel，优先合并访存 |

## 这个页面应该和什么一起看

- [Kernel 选择策略](/zh/architecture/kernel-selection)
- [内存布局](/zh/architecture/memory-layout)
- [性能方法学](/zh/performance/methodology)
