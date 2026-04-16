---
layout: default
title: 性能优化
---

# 性能优化指南

本文档详细介绍 GPU SpMV 库的性能优化策略、Kernel 选择机制、带宽优化技巧和基准测试方法。

---

## 目录

- [性能概览](#性能概览)
- [Kernel 选择策略](#kernel-选择策略)
- [带宽优化技术](#带宽优化技术)
- [基准测试指南](#基准测试指南)
- [性能调优建议](#性能调优建议)
- [常见问题](#常见问题)

---

## 性能概览

### SpMV 的性能瓶颈

稀疏矩阵向量乘法是典型的**带宽受限**（Memory-Bound）操作：

```
计算量: 2 × nnz 次浮点运算 (每次乘加)
数据传输: ~20 × nnz 字节 (values + indices + pointers + vectors)

算术强度: ~0.1 FLOP/Byte  (远低于 GPU 峰值需求)
```

因此，优化 SpMV 的关键在于**最大化内存带宽利用率**。

### 性能目标

| 指标 | 目标 | 说明 |
|------|------|------|
| 带宽利用率 | > 60% | 相对于 GPU 理论峰值带宽 |
| GFLOPS | 与带宽成比例 | GFLOPS = 2 × nnz / (time × 10^9) |
| 可扩展性 | 线性增长 | 性能随 nnz 近似线性增长 |

---

## Kernel 选择策略

SpMV 的性能高度依赖矩阵的非零元分布。本库提供自动 Kernel 选择机制，根据矩阵统计特征选取最优实现。

### 决策树

```
                    矩阵特征分析
                         │
                         ▼
         ┌───────────────────────────────┐
         │   avg_nnz_per_row < 4 ?       │
         └───────────────────────────────┘
                 │           │
                Yes          No
                 │           │
                 ▼           ▼
         ┌───────────┐  ┌───────────────────┐
         │   Scalar  │  │  skewness < 10 ?  │
         │    CSR    │  └───────────────────┘
         └───────────┘          │           │
                               Yes          No
                                │           │
                                ▼           ▼
                        ┌───────────┐ ┌───────────┐
                        │  Vector   │ │   Merge   │
                        │    CSR    │ │   Path    │
                        └───────────┘ └───────────┘
```

### Kernel 特性对比

| Kernel | 策略 | 适用场景 | 优势 | 劣势 |
|--------|------|----------|------|------|
| **Scalar CSR** | 一个线程/行 | 极稀疏矩阵 | 实现简单，无同步 | 负载不均 |
| **Vector CSR** | 一个 Warp/行 | 均匀分布 | Warp 归约高效 | 长行瓶颈 |
| **Merge Path** | 均匀工作量划分 | 高度不均匀 | 完美负载均衡 | 实现复杂 |
| **ELL** | Column-major | 行长度接近 | 访存完全合并 | 内存浪费 |

### Scalar CSR Kernel

```cpp
// 一个线程处理一行
__global__ void spmv_csr_scalar(int num_rows, int* row_ptrs, 
                                int* col_indices, float* values,
                                float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}
```

**适用场景：**
- `avg_nnz_per_row < 4`：每行非零元素很少
- 极稀疏矩阵（如网页链接图）

**性能特点：**
- 线程完全独立，无同步开销
- 短行时效率最高
- 长行会造成 Warp 空闲

### Vector CSR Kernel

```cpp
// 一个 Warp (32 线程) 协作处理一行
__global__ void spmv_csr_vector(int num_rows, int* row_ptrs,
                                int* col_indices, float* values,
                                float* x, float* y) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    if (warp_id < num_rows) {
        float sum = 0.0f;
        
        // Warp 内线程协作处理一行
        for (int j = row_ptrs[warp_id] + lane_id; 
             j < row_ptrs[warp_id + 1]; j += 32) {
            sum += values[j] * x[col_indices[j]];
        }
        
        // Warp 级归约 (使用 shuffle 指令，无 bank conflict)
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) y[warp_id] = sum;
    }
}
```

**适用场景：**
- `avg_nnz_per_row >= 4` 且 `skewness < 10`
- 行长度中等且分布均匀

**性能特点：**
- Warp 内并行处理长行
- Shuffle 归约避免 Shared Memory bank conflict
- 合并访问模式良好

### Merge Path Kernel

Merge Path 算法将行指针和非零元序列视为两条有序路径，通过二分搜索找到均匀分割点。

```
行指针序列:   [0, 2, 5, 7, 10]    (4 行，共 10 个非零元)
非零元序列:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Merge Path 视角:
         row:  0   1   2   3   4
              [0]─[2]─[5]─[7]─[10]
               │╲  │╲  │╲  │╲
              [0][1][2][3][4][5][6][7][8][9]  (nz)
               0   1   2   3   4   5   6   7   8   9

均匀划分: 对角线切割，每个线程处理相同数量的工作
```

**适用场景：**
- `skewness >= 10`：行长度差异极大
- 幂律分布图（如社交网络）

**性能特点：**
- 完美的负载均衡
- 每个线程处理相同数量的 (row, nz) 对
- 使用原子操作累加跨线程的部分和

### ELL Kernel

```cpp
// Column-major 存储，天然合并访问
__global__ void spmv_ell(int num_rows, int max_nnz_per_row,
                         int* col_indices, float* values,
                         float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_nnz_per_row; k++) {
            int idx = k * num_rows + row;  // Column-major 索引
            int col = col_indices[idx];
            if (col >= 0) {  // -1 表示填充
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
```

**适用场景：**
- 行长度接近 `max_nnz_per_row`
- ELL 格式矩阵

**性能特点：**
- 相邻线程访问连续内存地址（完全合并）
- 无条件分支（填充使用 -1 列索引）
- 存储效率取决于填充比例

---

## 带宽优化技术

### Column-Major 存储

ELL 格式的 Column-major 存储使得相邻线程访问连续内存地址：

```
Row-major 访问模式 (差):
线程:   T0      T1      T2
        ↓       ↓       ↓
地址: [row0,k0][row1,k0][row2,k0]  ← 不连续！
       [base+0] [base+max_nnz] [base+2*max_nnz]

Column-major 访问模式 (好):
线程:   T0      T1      T2
        ↓       ↓       ↓
地址: [row0,k0][row1,k0][row2,k0]  ← 连续！
       [base+0] [base+1]   [base+2]
```

### 纹理缓存

对输入向量 `x` 的访问是随机的（由列索引决定）。启用纹理缓存可以：

1. 利用 GPU 纹理缓存的空间局部性
2. 减少全局内存访问延迟
3. 对大规模向量效果显著

```cpp
SpMVConfig config = spmv_auto_config(csr);
config.use_texture = true;  // 启用纹理缓存

// 或自动启用：num_cols > 10000 时自动开启
```

**适用场景：**
- `num_cols > 10000`：大规模向量
- 随机访问模式明显

**注意事项：**
- 使用 `SpMVExecutionContext` 复用纹理对象
- 避免频繁创建/销毁纹理对象

### Warp 级归约

Vector CSR 使用 shuffle 指令进行归约，避免 Shared Memory bank conflict：

```cpp
// 传统 Shared Memory 归约 (有 bank conflict 风险)
__shared__ float sdata[32];
sdata[lane_id] = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    sdata[lane_id] += sdata[lane_id + offset];  // 可能冲突
}

// Shuffle 归约 (无 bank conflict)
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);  // 完全并行
}
```

---

## 基准测试指南

### 运行基准测试

```bash
# 构建后运行
cmake --preset release && cmake --build --preset release
./build/spmv_benchmark
```

### 输出示例

```
GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Memory: 10240 MB
Memory Bandwidth: 760.3 GB/s

========================================
SpMV Benchmark
========================================
Matrix: 1000x1000, NNZ: 50000, Density: 0.05

Scalar CSR:
  Avg time: 0.042 ms
  Min time: 0.038 ms
  Max time: 0.051 ms
  Stddev: 0.003 ms
  GFLOPS: 2381.0
  Bandwidth: 125.3 GB/s

Vector CSR:
  Avg time: 0.031 ms
  GFLOPS: 3225.8
  Bandwidth: 169.5 GB/s

Merge Path:
  Avg time: 0.035 ms
  GFLOPS: 2857.1
  Bandwidth: 150.2 GB/s

GPU vs CPU Comparison:
  GPU time: 0.031 ms
  CPU time: 2.450 ms
  Speedup: 79.0x
```

### 性能指标说明

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `avg_time_ms` | 平均执行时间 | 排除预热后的多次运行平均 |
| `gflops` | `2 × nnz / (time × 10^9)` | 每秒浮点运算次数 |
| `bandwidth_gb_s` | `bytes / (time × 10^9)` | 实际内存带宽 |
| `efficiency` | `achieved / theoretical` | 带宽利用率 |

### 字节计算公式

```cpp
// CSR 格式传输字节数
size_t csr_bytes = 0;
csr_bytes += nnz * sizeof(float);           // values
csr_bytes += nnz * sizeof(int);             // col_indices
csr_bytes += (num_rows + 1) * sizeof(int);  // row_ptrs
csr_bytes += num_cols * sizeof(float);      // x 向量
csr_bytes += num_rows * sizeof(float);      // y 向量

// ELL 格式传输字节数
size_t ell_size = num_rows * max_nnz_per_row;
size_t ell_bytes = 0;
ell_bytes += ell_size * sizeof(float);      // values
ell_bytes += ell_size * sizeof(int);        // col_indices
ell_bytes += num_cols * sizeof(float);      // x 向量
ell_bytes += num_rows * sizeof(float);      // y 向量
```

---

## 性能调优建议

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
    // 更新输入向量
    d_x.copyFromHost(new_x.data(), cols);
    
    // 执行 SpMV
    spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
    
    // 使用结果
    d_y.copyToHost(result.data(), rows);
}
```

### 4. 复用执行上下文

```cpp
// 多次调用复用纹理对象
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < 100; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
// 纹理对象在 context 析构时自动销毁
```

### 5. 检查带宽利用率

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);

float peak = get_gpu_peak_bandwidth();
float efficiency = result.bandwidth_gb_s / peak;

printf("Bandwidth utilization: %.1f%%\n", efficiency * 100);

if (efficiency < 0.5) {
    printf("Warning: Low bandwidth utilization. Consider:\n");
    printf("  - Using ELL format for uniform row lengths\n");
    printf("  - Enabling texture cache for large vectors\n");
    printf("  - Checking memory access patterns\n");
}
```

---

## 常见问题

### Q: 为什么带宽利用率低于预期？

**可能原因：**
1. **行长度差异大**：使用 Merge Path Kernel
2. **向量规模小**：GPU 未充分利用
3. **随机访问过多**：启用纹理缓存
4. **内存对齐问题**：检查数据布局

### Q: 如何选择最优 Block 大小？

```cpp
// 默认 256 适合大多数场景
// 可根据 GPU 架构调整：
// - 较少寄存器使用：增大 block_size
// - 较多共享内存：减小 block_size
config.block_size = 256;  // 推荐值
```

### Q: CPU 比 GPU 快怎么办？

对于小矩阵，GPU 开销可能超过收益：

```cpp
// 根据矩阵大小选择执行路径
if (csr->nnz < 10000) {
    spmv_cpu_csr(csr, x, y);  // CPU 更快
} else {
    spmv_csr(csr, d_x, d_y, &config, cols);  // GPU 更快
}
```

### Q: 如何处理 GPU 内存不足？

```cpp
// 分块处理大矩阵
int chunk_size = 1000000;  // 每块处理 100 万非零元素
for (int chunk = 0; chunk < num_chunks; chunk++) {
    // 处理当前块
    process_chunk(csr, chunk, chunk_size);
}
```

### Q: Merge Path 何时比 Vector CSR 更优？

当 `skewness >= 10` 时，Merge Path 通常更优：

```cpp
CSRStats stats = csr_compute_stats(csr);
// skewness = max_nnz_per_row / (min_nnz_per_row + 1)

if (stats.skewness >= 10) {
    // 行长度差异大，使用 Merge Path
    config.kernel_type = SpMVConfig::MERGE_PATH;
} else if (stats.avg_nnz_per_row >= 4) {
    // 行长度均匀，使用 Vector CSR
    config.kernel_type = SpMVConfig::VECTOR_CSR;
}
```
