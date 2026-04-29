---
layout: default
title: 性能优化
nav_order: 5
permalink: /performance
lang: zh
---

<p align="right">
  <a href="performance.en">🇺🇸 English</a>
</p>

# 性能优化
{: .no_toc }

调优策略、基准测试和最佳实践。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 性能概览

GPU SpMV 通过智能 Kernel 调度实现高性能：

- **目标带宽利用率**: > 60%
- **实测峰值**: 70-75% (RTX 3090)
- **性能可扩展性**: 随矩阵规模线性增长

---

## 基准测试结果

### NVIDIA RTX 3090 (Ampere, 936 GB/s)

| 矩阵规模 | 非零元素 | Kernel | 带宽利用率 |
|:--------:|:--------:|:-------|:----------:|
| 10K × 10K | 500K | Vector CSR | **70.2%** |
| 100K × 100K | 5M | Merge Path | **71.5%** |
| 1M × 1M | 50M | Merge Path | **70.8%** |

### 不同 GPU 架构对比

| GPU 架构 | 理论带宽 | 实测利用率 |
|:---------|:--------:|:----------:|
| Volta (V100) | 900 GB/s | ~65% |
| Turing (RTX 2080) | 448 GB/s | ~68% |
| Ampere (RTX 3090) | 936 GB/s | ~70% |
| Ada Lovelace (RTX 4090) | 1008 GB/s | ~72% |

---

## Kernel 选择策略

```
矩阵特征分析
       │
       ├── 平均非零元/行 < 4 ──→ Scalar CSR (单线程/行)
       │
       ├── 平均非零元/行 ≥ 4
       │       │
       │       ├── 偏度 < 10 ──→ Vector CSR (warp/行)
       │       │
       │       └── 偏度 ≥ 10 ──→ Merge Path (负载均衡)
       │
       └── ELL 格式 ────────→ ELL Kernel (合并访存)
```

### Kernel 类型

| Kernel | 适用场景 | 带宽利用率 |
|:-------|:---------|:----------:|
| Scalar CSR | 极稀疏矩阵 | ~40-50% |
| Vector CSR | 均匀分布 | ~65-75% |
| Merge Path | 高度倾斜 | ~70-80% |
| ELL | 行长度均匀 | ~80-90% |

---

## 优化指南

### 1. 使用自动配置（推荐）

```cpp
// 库自动选择最优 Kernel
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

### 2. 格式选择

```cpp
// 行长度均匀时，转换为 ELL 格式
if (row_length_variance < 0.2) {
    ELLMatrix* ell = ell_create(rows, cols, max_nnz_per_row);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
    // ELL 通常性能更好
}
```

### 3. 内存优化

```cpp
// ✅ 使用 RAII 自动管理
CudaBuffer<float> buffer(n);

// ❌ 避免手动分配
float* ptr;
cudaMalloc(&ptr, n * sizeof(float));
// 容易忘记 cudaFree
```

---

## 性能分析工具

```bash
# Nsight Systems 整体分析
nsys profile ./spmv_benchmark

# Nsight Compute 详细 Kernel 分析
ncu --kernel-name spmv ./spmv_benchmark

# 内置基准测试
./build-release/spmv_benchmark
```

---

## 常见问题

### 性能不达预期？

检查清单：

1. ✅ 使用了 `spmv_auto_config()`
2. ✅ 矩阵已传输到 GPU (`csr_to_gpu`)
3. ✅ 输入向量在 GPU 上
4. ✅ 矩阵规模足够大 (>10K 非零元)

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>完整测试数据见 <a href="https://github.com/LessUp/gpu-spmv/tree/main/benchmarks">benchmarks/</a> 目录</p>
</div>
