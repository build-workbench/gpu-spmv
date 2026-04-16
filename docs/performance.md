---
layout: default
title: 性能优化
lang: zh
---

<p align="right">
  <a href="performance.en.html">🇺🇸 English</a>
</p>

# 🚀 性能优化指南

本指南详细介绍 GPU SpMV 库的性能优化策略、Kernel 选择机制和基准测试最佳实践。

---

## 目录

- [性能概览](#性能概览)
- [Kernel 选择](#kernel-选择)
- [内存带宽](#内存带宽)
- [基准测试](#基准测试)
- [优化技巧](#优化技巧)

---

## 性能概览

### 带宽受限特性

SpMV 是典型的**内存带宽受限**操作：

```
计算量: 2 × nnz 次浮点运算 (每非零元素一次乘加)
数据量: ~20 × nnz 字节 (数值 + 索引 + 向量)

算术强度: ~0.1 FLOP/字节 (远低于 GPU 能力)
```

因此，SpMV 优化的重点是**最大化内存带宽利用率**。

### 性能目标

| 指标 | 目标 | 说明 |
|:-----|:-----|:-----|
| 带宽利用率 | > 60% | 相对于 GPU 理论峰值 |
| GFLOPS | 与带宽成比例 | GFLOPS = 2 × nnz / (时间 × 10^9) |
| 可扩展性 | 线性 | 性能随 nnz 近似线性增长 |

---

## Kernel 选择

SpMV 性能高度依赖非零元分布。本库提供智能 Kernel 选择机制。

### 决策树

```
                 矩阵特征分析
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────┴────┐              ┌─────┴──────┐
    │  格式   │              │  统计特征   │
    └────┬────┘              └─────┬──────┘
         │                         │
    ELL ─┼─→ ELL Kernel    ┌──────┴──────┐
         │                 │             │
        CSR          avg_nnz < 4?   skewness < 10?
                            │             │
                          Yes            No
                            │             │
                       ┌────┴────┐   ┌────┴────┐
                       │ Scalar  │   │ Vector  │
                       │   CSR   │   │   CSR   │
                       └─────────┘   └────┬────┘
                                          │
                                         No
                                          │
                                     ┌────┴────┐
                                     │  Merge  │
                                     │  Path   │
                                     └─────────┘
```

### Kernel 对比

| Kernel | 并行策略 | 同步开销 | 负载均衡 | 最佳场景 |
|:-------|:---------|:--------:|:--------:|:---------|
| **Scalar CSR** | 1 线程/行 | 无 | 差 | avg_nnz < 4 |
| **Vector CSR** | 1 Warp/行 | Warp 归约 | 中等 | 均匀分布 |
| **Merge Path** | 均匀划分 | 原子操作 | 完美 | 极度不均匀 |
| **ELL** | Column-major | 无 | - | ELL 格式 |

---

## 内存带宽

### 合并访问

ELL Column-major 存储实现完全合并访问：

```
行优先 (差):
线程:  T0      T1      T2
       ↓       ↓       ↓
地址: [r0,k0][r1,k0][r2,k0]  ← 不连续

列优先 (好):
线程:  T0      T1      T2
       ↓       ↓       ↓
地址: [r0,k0][r1,k0][r2,k0]  ← 连续!
       [base+0] [base+1] [base+2]
```

### 纹理缓存

输入向量 `x` 的随机访问（由列索引决定）受益于纹理缓存：

```cpp
SpMVConfig config = spmv_auto_config(csr);
config.use_texture = true;  // 启用纹理缓存

// 或当 num_cols > 10000 时自动启用
```

**使用场景:**
- `num_cols > 10000`: 大规模向量
- 明显的随机访问模式

**最佳实践:**
- 使用 `SpMVExecutionContext` 复用纹理对象
- 避免频繁创建/销毁

### Warp Shuffle 归约

Vector CSR 使用 shuffle 指令进行归约，避免共享内存 bank conflict：

```cpp
// 传统共享内存归约（可能有 bank conflict）
__shared__ float sdata[32];
sdata[lane_id] = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    sdata[lane_id] += sdata[lane_id + offset];  // 可能冲突
}

// Shuffle 归约（无 bank conflict）
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);  // 完全并行
}
```

---

## 基准测试

### 运行基准测试

```bash
# 构建并运行
cmake --preset release && cmake --build --preset release
./build-release/spmv_benchmark
```

### 输出示例

```
========================================
GPU SpMV Benchmark
========================================
GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Memory: 10240 MB
Memory Bandwidth: 760.3 GB/s

Matrix: 1000x1000, NNZ: 50000, Density: 0.05

Scalar CSR:
  平均时间: 0.042 ms
  GFLOPS: 2381.0
  带宽: 125.3 GB/s (16.5%)

Vector CSR:
  平均时间: 0.031 ms
  GFLOPS: 3225.8
  带宽: 169.5 GB/s (22.3%)

Merge Path:
  平均时间: 0.035 ms
  GFLOPS: 2857.1
  带宽: 150.2 GB/s (19.8%)
```

### 指标说明

| 指标 | 计算方式 | 含义 |
|:-----|:---------|:-----|
| 平均时间 | 多次运行平均 | 排除预热 |
| GFLOPS | `2 × nnz / (时间 × 10^9)` | 浮点运算吞吐量 |
| 带宽 | `字节数 / (时间 × 10^9)` | 实际内存带宽 |
| 利用率 | `实际 / 理论峰值` | 带宽利用率 |

---

## 优化技巧

### 1. 选择合适的矩阵格式

```cpp
// 计算 ELL 存储效率
float ell_efficiency = (float)nnz / (num_rows * max_nnz_per_row);

if (ell_efficiency > 0.8) {
    // 使用 ELL 格式
    ELLMatrix* ell = ell_create(0, 0, 0);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
} else {
    // 使用 CSR 格式
    csr_to_gpu(csr);
}
```

### 2. 启用纹理缓存

```cpp
// 大规模向量启用纹理缓存
SpMVConfig config = spmv_auto_config(csr);
if (csr->num_cols > 10000) {
    config.use_texture = true;
}
```

### 3. 复用 GPU 内存

```cpp
// 避免重复分配
CudaBuffer<float> d_x(cols);
CudaBuffer<float> d_y(rows);

for (int iter = 0; iter < iterations; iter++) {
    d_x.copyFromHost(new_x.data(), cols);
    spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
    d_y.copyToHost(result.data(), rows);
}
```

### 4. 复用执行上下文

```cpp
// 跨调用复用纹理对象
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < 100; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
// 纹理对象随 context 析构自动释放
```

### 5. 检查带宽利用率

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);

float peak = get_gpu_peak_bandwidth();
float efficiency = result.bandwidth_gb_s / peak;

printf("带宽利用率: %.1f%%\n", efficiency * 100);

if (efficiency < 0.5) {
    printf("建议:\n");
    printf("  - 均匀行长的矩阵使用 ELL 格式\n");
    printf("  - 大规模向量启用纹理缓存\n");
}
```

---

<div align="center">

**[← 示例代码](examples)** · **[ 更新日志 →](changelog)**

</div>
