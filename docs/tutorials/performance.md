---
layout: default
title: 性能优化
nav_order: 5
has_children: false
lang: zh
---

<p align="right">
  <a href="performance.en">🇺🇸 English</a>
</p>

# 🚀 性能优化
{: .no_toc }

调优策略、基准测试和最佳实践。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 性能概述

GPU SpMV 库通过多种 Kernel 智能调度实现极致性能。

### 核心性能指标

| 指标 | 描述 | 目标值 |
|:-----|:-----|:------|
| **带宽利用率** | 实际内存带宽 / 理论峰值 | > 60% |
| **计算密度** | FLOPS / 字节访问 | 视矩阵而定 |
| **可扩展性** | 随矩阵规模增长的性能 | 线性增长 |

---

## Kernel 选择策略

### 1. Scalar CSR Kernel

**适用场景**: 极稀疏矩阵 (avg_nnz < 4)

```cpp
// 自动选择
SpMVConfig config = spmv_auto_config(csr);
// config.kernel_type == KernelType::SCALAR_CSR
```

**性能特征**:
- 每个线程处理一行
- 最小化线程间协调开销
- 适合非零元素极少的情况

**带宽利用率**: ~40-50%

### 2. Vector CSR Kernel

**适用场景**: 中等稀疏矩阵 (skewness < 10)

**性能特征**:
- 每个 warp 协作处理一行
- 合并访存模式
- 均衡的负载分配

**带宽利用率**: ~65-75%

### 3. Merge Path Kernel

**适用场景**: 高度倾斜矩阵 (skewness ≥ 10)

**性能特征**:
- 完美负载均衡
- 二分查找分割点
- 自适应矩阵特征

**带宽利用率**: ~70-80%

### 4. ELL Kernel

**适用场景**: ELL 格式矩阵

**性能特征**:
- 完全合并访存
- 列主序存储
- 最高带宽利用率

**带宽利用率**: ~80-90%

---

## 性能调优指南

### 1. 自动配置（推荐）

```cpp
// 让库自动选择最优 Kernel
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

**优势**:
- 无需手动调优
- 根据矩阵特征智能选择
- 适合大多数场景

### 2. 手动选择 Kernel

```cpp
// 针对特定场景手动选择
SpMVConfig config;
config.kernel_type = KernelType::MERGE_PATH;
config.auto_select = false;

SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

**适用场景**:
- 已知矩阵特征稳定
- 需要极致性能
- 自动选择结果不理想

### 3. 格式转换

```cpp
// CSR -> ELL 转换
ELLMatrix* ell = ell_create(num_rows, num_cols, max_nnz_per_row);
ell_from_csr(ell, csr);
ell_to_gpu(ell);

// ELL 格式通常性能更好
SpMVResult result = spmv_ell(ell, d_x, d_y, n);
```

**何时转换**:
- 矩阵行长度均匀
- 每行非零元素数差异 < 20%
- 追求极致性能

---

## 基准测试

### 运行基准测试

```cpp
#include <spmv/benchmark.h>

BenchmarkConfig config;
config.iterations = 100;      // 迭代 100 次
config.warmup = true;         // 预热
config.print_details = true;  // 详细信息

spmv_benchmark(csr, &config);
```

### 输出示例

```
=== GPU SpMV Benchmark ===
Matrix: 10000 x 10000, nnz = 500000
Kernel: Vector CSR
Iterations: 100 (10 warmup)

Results:
  Avg:  2.34 ms
  Min:  2.12 ms
  Max:  2.89 ms
  Std:  0.15 ms

Bandwidth: 68.5 GB/s (70.2% of peak)
GFLOPS: 42.8
```

### 自定义基准测试

```cpp
#include <spmv/spmv.h>
#include <chrono>

void custom_benchmark(const CSRMatrix* csr, 
                     const float* d_x, 
                     float* d_y, 
                     int n,
                     int iterations) {
    // 预热
    SpMVConfig config = spmv_auto_config(csr);
    for (int i = 0; i < 5; i++) {
        spmv_csr(csr, d_x, d_y, &config, n);
    }
    
    // 正式测试
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        spmv_csr(csr, d_x, d_y, &config, n);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count();
    
    printf("Avg: %.3f ms\n", duration / 1000.0 / iterations);
}
```

---

## 性能优化最佳实践

### 1. 内存优化

```cpp
// ✅ 推荐: 使用 RAII
void process() {
    CudaBuffer<float> d_x(1000000);
    CudaBuffer<float> d_y(1000000);
    // 自动管理生命周期
}

// ❌ 避免: 手动管理
void process() {
    float *d_x, *d_y;
    cudaMalloc(&d_x, 1000000 * sizeof(float));
    cudaMalloc(&d_y, 1000000 * sizeof(float));
    // 容易忘记 cudaFree
    cudaFree(d_x);
    cudaFree(d_y);
}
```

### 2. 执行上下文复用

```cpp
// ✅ 推荐: 复用上下文
SpMVExecutionContext ctx;
for (int i = 0; i < 100; i++) {
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
    // 纹理对象和缓存配置被复用
}

// ❌ 避免: 每次创建
for (int i = 0; i < 100; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
    // 重复创建纹理对象
}
```

### 3. 批量处理

```cpp
// ✅ 推荐: 批量处理多个矩阵
void process_batch(const std::vector<CSRMatrix*>& matrices) {
    for (auto* csr : matrices) {
        csr_to_gpu(csr);
        SpMVConfig config = spmv_auto_config(csr);
        spmv_csr(csr, d_x, d_y, &config, csr->num_rows);
    }
}

// ❌ 避免: 逐个处理
for (int i = 0; i < 100; i++) {
    CSRMatrix* csr = load_matrix(i);
    csr_to_gpu(csr);
    spmv_csr(csr, d_x, d_y, &config, n);
    csr_destroy(csr);
}
```

### 4. 数据传输优化

```cpp
// ✅ 推荐: 异步传输
cudaMemcpyAsync(d_x.data(), h_x.data(), 
                n * sizeof(float), 
                cudaMemcpyHostToDevice, 
                stream);

// ❌ 避免: 同步传输阻塞
cudaMemcpy(d_x.data(), h_x.data(), 
           n * sizeof(float), 
           cudaMemcpyHostToDevice);
```

---

## 性能基准数据

### NVIDIA RTX 3090 (Ampere) 测试结果

| 矩阵规模 | 非零元素 | Kernel | 时间 (ms) | 带宽 (GB/s) | 利用率 |
|:--------:|:--------:|:-------|:---------:|:-----------:|:------:|
| 10K × 10K | 500K | Vector CSR | 2.34 | 68.5 | 70.2% |
| 50K × 50K | 2.5M | Merge Path | 11.8 | 71.2 | 72.9% |
| 100K × 100K | 5M | Merge Path | 23.5 | 69.8 | 71.5% |
| 500K × 500K | 25M | Merge Path | 118.3 | 70.5 | 72.2% |
| 1M × 1M | 50M | Merge Path | 235.7 | 69.1 | 70.8% |

### 不同 GPU 架构对比

| GPU 架构 | 代表型号 | 理论带宽 | 实际利用率 |
|:---------|:---------|:--------:|:----------:|
| Volta | V100 | 900 GB/s | ~65% |
| Turing | RTX 2080 | 448 GB/s | ~68% |
| Ampere | RTX 3090 | 936 GB/s | ~70% |
| Ada Lovelace | RTX 4090 | 1008 GB/s | ~72% |

---

## 故障排查

### 性能低下的常见原因

1. **未使用自动配置**
   ```cpp
   // ❌ 错误
   SpMVConfig config;
   config.kernel_type = KernelType::SCALAR_CSR;  // 手动选择了低效 Kernel
   ```

2. **数据未传输到 GPU**
   ```cpp
   // ❌ 错误
   csr_to_gpu(csr);  // 忘记调用
   spmv_csr(csr, d_x, d_y, &config, n);  // 在 CPU 数据上执行
   ```

3. **矩阵规模过小**
   ```cpp
   // 警告: 100x100 矩阵 overhead 占比高
   CSRMatrix* csr = csr_create(100, 100, 500);
   ```

4. **频繁的内存分配**
   ```cpp
   // ❌ 错误: 循环中分配
   for (int i = 0; i < 100; i++) {
       CudaBuffer<float> buf(1000);  // 每次迭代都分配
   }
   ```

### 性能分析工具

```bash
# 使用 nvprof 分析
nvprof ./spmv_benchmark

# 使用 Nsight Systems
nsys profile ./spmv_benchmark

# 使用 Nsight Compute
ncu --kernel-name spmv ./spmv_benchmark
```

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>完整性能数据见 <a href="https://github.com/LessUp/gpu-spmv/tree/main/benchmarks">benchmarks/</a> 目录</p>
</div>
