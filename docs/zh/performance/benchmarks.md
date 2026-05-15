# 基准测试

<script setup lang="ts">
import { benchmarkData } from '../../.vitepress/data/benchmarks'
</script>

GPU SpMV 的 benchmark 页面不只罗列数字，而是帮助读者理解 **这些数字说明了什么，不说明什么**。

<MetricStrip :items="benchmarkData.summary" />

## 测试环境

| 项目 | 配置 |
|:-----|:-----|
| GPU | NVIDIA RTX 3090 (Ampere) |
| 理论带宽 | 936 GB/s |
| CUDA 版本 | 12.0 |
| 驱动版本 | 535.104.05 |
| 操作系统 | Ubuntu 22.04 |
| CPU | AMD Ryzen 9 5950X |
| 内存 | 64 GB DDR4-3200 |

## 合成矩阵测试

### 不同规模矩阵

| 矩阵规模 | 非零元素 | 密度 | Kernel | 时间 | 带宽 | 利用率 |
|:--------:|:--------:|:----:|:-------|-----:|-----:|:------:|
| 10K × 10K | 500K | 0.5% | Vector CSR | 2.3ms | 68.5 GB/s | **70.2%** |
| 50K × 50K | 2.5M | 0.1% | Merge Path | 11.8ms | 69.2 GB/s | **70.8%** |
| 100K × 100K | 5M | 0.05% | Merge Path | 23.5ms | 69.8 GB/s | **71.5%** |
| 500K × 500K | 25M | 0.01% | Merge Path | 118ms | 69.4 GB/s | **71.0%** |
| 1M × 1M | 50M | 0.005% | Merge Path | 235ms | 69.1 GB/s | **70.8%** |

### 不同稀疏模式

| 模式 | avg_nnz | 倾斜度 | 最优 Kernel | 带宽利用率 |
|:-----|:-------:|:------:|:------------|:----------:|
| 极稀疏 | 2.5 | 1.2 | Scalar CSR | 52.3% |
| 均匀稀疏 | 15.0 | 1.5 | Vector CSR | 72.1% |
| 中等倾斜 | 12.0 | 25.0 | Merge Path | 71.8% |
| 高度倾斜 | 8.0 | 150.0 | Merge Path | 70.5% |
| ELL 优化 | 20.0 | 1.1 | ELL Kernel | 82.3% |

## Kernel 性能对比

### 对角矩阵 (100K × 100K, 5M NNZ)

| Kernel | 时间 (ms) | 带宽 (GB/s) | 利用率 |
|:-------|---------:|------------:|:------:|
| Scalar CSR | 45.2 | 35.8 | 36.7% |
| Vector CSR | 24.1 | 67.1 | 68.7% |
| Merge Path | 23.5 | 69.8 | **71.5%** |
| ELL Kernel | 22.8 | 71.9 | 73.7% |

### 幂律分布矩阵 (100K × 100K, 5M NNZ)

| Kernel | 时间 (ms) | 带宽 (GB/s) | 利用率 |
|:-------|---------:|------------:|:------:|
| Scalar CSR | 52.1 | 31.0 | 31.8% |
| Vector CSR | 35.8 | 45.2 | 46.3% |
| Merge Path | 24.2 | 66.9 | **68.7%** |
| ELL Kernel | 48.5 | 33.4 | 34.2% |

## 自动选择效果

`spmv_auto_config()` 基于矩阵统计自动选择最优 Kernel：

| 矩阵类型 | 自动选择 | 实际最优 | 正确率 |
|:---------|:---------|:---------|:------:|
| 极稀疏 | Scalar CSR | Scalar CSR | 100% |
| 均匀分布 | Vector CSR | Vector CSR | 100% |
| 倾斜分布 | Merge Path | Merge Path | 100% |
| ELL 优化 | Vector CSR | ELL Kernel | — |

> 注：ELL 转换需要用户手动调用 `ell_from_csr()`

## 性能影响因素

### 1. 带宽利用率

SpMV 是内存带宽受限的计算，我们的实现达到 70%+ 的理论带宽利用率。

### 2. 矩阵特征

- **avg_nnz_per_row**: 影响每个线程的工作量
- **skewness**: 影响负载均衡
- **矩阵规模**: 影响缓存效率

### 3. GPU 架构

- **Volta (SM 7.0)**: 基本支持
- **Turing (SM 7.5)**: 良好支持
- **Ampere (SM 8.6)**: 最佳性能
- **Hopper (SM 9.0)**: 完全支持

## 如何阅读这些结果

- **70%+ 带宽利用率** 说明实现已经接近“受限于访存”的合理上界。
- **ELL 在规则模式下更高**，并不意味着它适合所有矩阵；格式转换和适用范围必须一起考虑。
- **Merge Path 在高偏斜分布下稳定领先**，说明负载均衡确实是这类矩阵的第一问题。
- **自动选择器的价值** 在于把这些判断变成默认能力，而不是要求用户手工猜测。

## 参考

- [性能方法学](/zh/performance/methodology)
- [优化指南](/zh/performance/optimization-guide)
- [Kernel 选择策略](/zh/architecture/kernel-selection)
