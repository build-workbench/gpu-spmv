---
layout: default
title: API 参考
---

# API 参考

## 稀疏矩阵格式

### CSR (Compressed Sparse Row)

```cpp
#include "spmv/csr_matrix.h"
```

| 函数 | 说明 |
|------|------|
| `csr_create(rows, cols, nnz)` | 创建 CSR 矩阵 |
| `csr_destroy(csr)` | 释放 CSR 矩阵及其 GPU 镜像 |
| `csr_from_dense(csr, data, rows, cols)` | 从行优先稠密矩阵重建 CSR 数据 |
| `csr_to_dense(csr, dense)` | 将 CSR 转回稠密矩阵 |
| `csr_get_element(csr, row, col)` | 查询单个元素 |
| `csr_to_gpu(csr)` | 上传 CSR 数据到 GPU |
| `csr_from_gpu(csr)` | 从 GPU 拷回主机 |
| `csr_free_gpu(csr)` | 释放 GPU 端镜像 |
| `csr_serialize(csr, filename)` | 序列化到文件 |
| `csr_deserialize(csr, filename)` | 从文件反序列化 |
| `csr_compute_stats(csr)` | 计算平均每行非零数、偏斜度等统计信息 |

#### 数据结构

```cpp
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;

    float* values;
    int* col_indices;
    int* row_ptrs;

    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

### ELL (ELLPACK)

```cpp
#include "spmv/ell_matrix.h"
```

| 函数 | 说明 |
|------|------|
| `ell_create(rows, cols, max_nnz_per_row)` | 创建 ELL 矩阵 |
| `ell_destroy(ell)` | 释放 ELL 矩阵及其 GPU 镜像 |
| `ell_from_dense(ell, data, rows, cols)` | 从稠密矩阵重建 ELL 数据 |
| `ell_from_csr(ell, csr)` | 从 CSR 转换为 ELL |
| `ell_to_dense(ell, dense)` | 将 ELL 转回稠密矩阵 |
| `ell_get_element(ell, row, col)` | 查询单个元素 |
| `ell_to_gpu(ell)` | 上传 ELL 数据到 GPU |
| `ell_from_gpu(ell)` | 从 GPU 拷回主机 |
| `ell_free_gpu(ell)` | 释放 GPU 端镜像 |
| `ell_serialize(ell, filename)` | 序列化到文件 |
| `ell_deserialize(ell, filename)` | 从文件反序列化 |
| `ell_index(row, k, num_rows)` | 计算 Column-major 存储下标 |

#### 数据结构

```cpp
struct ELLMatrix {
    int num_rows;
    int num_cols;
    int max_nnz_per_row;
    int nnz;

    float* values;
    int* col_indices;

    float* d_values;
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

## SpMV 接口

```cpp
#include "spmv/spmv.h"
```

### Kernel 类型

| `SpMVConfig::KernelType` | 适用场景 |
|--------------------------|----------|
| `SpMVConfig::SCALAR_CSR` | CSR 短行场景，一个线程处理一行 |
| `SpMVConfig::VECTOR_CSR` | CSR 较均匀场景，一个 warp 处理一行 |
| `SpMVConfig::MERGE_PATH` | CSR 高度不均匀场景，按工作量均衡划分 |
| `SpMVConfig::ELL_KERNEL` | ELL 格式专用 Kernel |

### 核心函数

```cpp
// CPU 参考实现
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

// 自动选择 CSR Kernel 配置
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* A,
                    const float* d_x,
                    float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* A,
                    const float* d_x,
                    float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1);
```

### `SpMVConfig`

```cpp
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,
        VECTOR_CSR,
        MERGE_PATH,
        ELL_KERNEL
    };

    KernelType kernel_type;
    int block_size;
    bool use_texture;

    SpMVConfig();
    SpMVConfig(KernelType kernel_type_, int block_size_, bool use_texture_);
};
```

- `kernel_type`：显式指定 Kernel 类型。
- `block_size`：CUDA block 大小，当前实现要求为 32 的倍数。
- `use_texture`：是否启用纹理缓存读取输入向量。

### `SpMVResult`

```cpp
struct SpMVResult {
    float* y;
    float elapsed_ms;
    float gflops;
    float bandwidth_gb_s;
    int error_code;
};
```

- `y`：输出向量指针，通常等于传入的 `d_y`。
- `error_code`：语义化错误码，取值见 `SpMVError`。

## RAII 工具

### `CudaBuffer`

```cpp
#include "spmv/cuda_buffer.h"

CudaBuffer<float> buf(1024);
buf.copyFromHost(host_ptr, count);
buf.copyToHost(host_ptr, count);
buf.memset(0);
buf.fill(value);
size_t n = buf.bytes();
float* ptr = buf.get();
```

### `CudaTimer`

```cpp
// 在 CUDA 实现内部用于计时
// 基于 cudaEvent 的 RAII 封装
```

### `ScopedTexture`

```cpp
// 在 CUDA 实现内部按需使用
// 负责纹理对象生命周期管理
```

## 基准测试

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig config;
config.num_warmup_runs = 5;
config.num_runs = 20;
config.compare_cpu = true;

BenchmarkResult csr_result = benchmark_csr(csr, x, &spmv_config, &config);
BenchmarkResult ell_result = benchmark_ell(ell, x, &config);
ComparisonResult comp = compare_gpu_cpu_csr(csr, x, &spmv_config, &config);

if (csr_result.error_code == static_cast<int>(SpMVError::SUCCESS)) {
    std::string json = benchmark_to_json(csr_result);
}
```

### 结果结构

```cpp
struct BenchmarkResult {
    std::string name;
    float execution_time_ms;
    float gflops;
    float bandwidth_gb_s;
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float stddev_time_ms;
    int num_runs;
    int error_code;
};
```

- `num_runs`：成功完成的采样次数。
- `error_code`：显式错误码；当输入非法、GPU 镜像缺失或内部 SpMV 失败时，会返回非成功值。

### 对比结果

```cpp
struct ComparisonResult {
    BenchmarkResult gpu_result;
    BenchmarkResult cpu_result;
    float speedup;
    int error_code;
};
```

## 带宽度量

```cpp
#include "spmv/bandwidth.h"

BandwidthMetrics csr_bw = compute_bandwidth_csr(csr, elapsed_ms);
BandwidthMetrics ell_bw = compute_bandwidth_ell(ell, elapsed_ms);
float peak = get_gpu_peak_bandwidth();
```

```cpp
struct BandwidthMetrics {
    float theoretical_bandwidth_gb_s;
    float achieved_bandwidth_gb_s;
    float efficiency;
};
```

## PageRank

```cpp
#include "spmv/pagerank.h"

PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;
config.max_iterations = 100;

PageRankResult result = pagerank(adj, &config);

if (result.error_code == static_cast<int>(SpMVError::SUCCESS)) {
    std::vector<TopKNode> top_k(10);
    pagerank_top_k(&result, adj->num_rows, 10, top_k.data());
}

pagerank_free(&result);
```

### `PageRankConfig`

```cpp
struct PageRankConfig {
    float damping_factor;
    float tolerance;
    int max_iterations;
};
```

### `PageRankResult`

```cpp
struct PageRankResult {
    float* ranks;
    int iterations;
    float final_residual;
    bool converged;
    int error_code;
};
```

- `ranks`：长度为 `num_nodes` 的主机端结果数组。
- `converged`：是否在 `max_iterations` 内达到收敛阈值。
- `error_code`：显式错误通道；调用方应先检查该字段，再消费 `ranks` 或调用 `pagerank_top_k`。

## 错误处理

```cpp
#include "spmv/common.h"

CUDA_CHECK_MALLOC(cudaMalloc(...));
CUDA_CHECK_MEMCPY(cudaMemcpy(...));
CUDA_CHECK(cudaMalloc(...));  // 向后兼容别名
```

| `SpMVError` | 含义 |
|-------------|------|
| `SpMVError::SUCCESS` | 成功 |
| `SpMVError::INVALID_DIMENSION` | 矩阵或向量维度不匹配 |
| `SpMVError::CUDA_MALLOC` | GPU 内存分配失败 |
| `SpMVError::CUDA_MEMCPY` | GPU / 主机内存拷贝失败 |
| `SpMVError::KERNEL_LAUNCH` | CUDA Kernel 启动或执行失败 |
| `SpMVError::INVALID_FORMAT` | 稀疏矩阵格式非法或 GPU 镜像缺失 |
| `SpMVError::FILE_IO` | 文件读写失败 |
| `SpMVError::OUT_OF_MEMORY` | 主机或设备内存不足 |
| `SpMVError::INVALID_ARGUMENT` | 参数非法 |
