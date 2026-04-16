---
layout: default
title: 性能优化
parent: 中文文档
nav_order: 5
lang: zh
---

# 🚀 性能优化
{: .no_toc }

性能优化策略和最佳实践。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 性能概览

SpMV 是典型的**内存带宽受限**操作：

```
计算量: 2 × nnz FLOPs
数据量: ~20 × nnz bytes
算术强度: ~0.1 FLOP/Byte
```

→ 优化关键是最大化内存带宽利用率

---

## Kernel 选择

| 条件 | 推荐 Kernel |
|:-----|:------------|
| `avg_nnz_per_row < 4` | Scalar CSR |
| `skewness < 10` | Vector CSR |
| `skewness >= 10` | Merge Path |
| ELL 格式 | ELL Kernel |

---

## 优化技巧

### 1. 启用纹理缓存

```cpp
SpMVConfig config = spmv_auto_config(csr);
if (csr->num_cols > 10000) {
    config.use_texture = true;
}
```

### 2. 复用 GPU 内存

```cpp
CudaBuffer<float> d_x(cols), d_y(rows);

for (int iter = 0; iter < iterations; iter++) {
    d_x.copyFromHost(new_x.data(), cols);
    spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
}
```

### 3. 检查带宽利用率

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);
float peak = get_gpu_peak_bandwidth();
float efficiency = result.bandwidth_gb_s / peak;
```

---

## 基准测试

```bash
./build-release/spmv_benchmark
```

预期输出：
```
GPU: NVIDIA GeForce RTX 3080
Memory Bandwidth: 760.3 GB/s

Matrix: 100000x100000, NNZ: 5000000
Avg time: 0.312 ms
Bandwidth: 495.2 GB/s (65.1%)
```
