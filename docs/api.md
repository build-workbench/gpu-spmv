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
| `csr_create(nnz, rows, cols)` | 创建 CSR 矩阵 |
| `csr_destroy(csr)` | 释放 CSR 矩阵 |
| `csr_from_dense(csr, data, rows, cols)` | 从稠密矩阵转换 |
| `csr_to_gpu(csr)` | 上传到 GPU |
| `csr_get_element(csr, row, col)` | 获取元素值 |

### ELL (ELLPACK)

```cpp
#include "spmv/ell_matrix.h"
```

| 函数 | 说明 |
|------|------|
| `ell_create(rows, max_nnz)` | 创建 ELL 矩阵 |
| `ell_destroy(ell)` | 释放 ELL 矩阵 |
| `ell_from_csr(ell, csr)` | 从 CSR 转换 |
| `ell_to_gpu(ell)` | 上传到 GPU |

## SpMV 接口

```cpp
#include "spmv/spmv.h"
```

### Kernel 类型

| Kernel | 适用场景 |
|--------|----------|
| `SPMV_SCALAR` | 短行（avg_nnz < 4） |
| `SPMV_VECTOR` | 均匀分布（skewness < 10） |
| `SPMV_MERGE_PATH` | 高度不均匀（skewness >= 10） |
| `SPMV_ELL` | ELL 格式 |

### 核心函数

```cpp
// 自动选择最优 Kernel
SpMVConfig spmv_auto_config(const CSRMatrix* csr);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* csr,
                    const float* d_x, float* d_y,
                    const SpMVConfig* config, int n);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* ell,
                    const float* d_x, float* d_y, int n);
```

### SpMVConfig

```cpp
struct SpMVConfig {
    SpMVKernelType kernel;   // Kernel 类型
    bool use_texture;        // 是否启用纹理缓存读取输入向量
};
```

## RAII 工具

### CudaBuffer

```cpp
#include "spmv/cuda_buffer.h"

CudaBuffer<float> buf(1024);        // 分配 1024 个 float
buf.copyFromHost(host_ptr, count);   // 主机 → GPU
buf.copyToHost(host_ptr, count);     // GPU → 主机
buf.memset(0);                       // 清零
buf.fill(value);                     // 填充
size_t n = buf.bytes();              // 字节数
float* ptr = buf.get();              // 获取原始指针
```

### CudaTimer

```cpp
// 在 spmv_kernels.cu 中自动使用
// RAII 封装 cudaEvent，构造时记录开始，析构时计算耗时
```

### ScopedTexture

```cpp
// 在 spmv_kernels.cu 中自动使用
// RAII 封装纹理对象，析构时自动销毁
```

## 基准测试

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig config;
config.num_warmup_runs = 5;
config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x, &spmv_config, &config);
// result.avg_time_ms, result.gflops, result.bandwidth_gb_s

std::string json = benchmark_to_json(result);
```

## 带宽度量

```cpp
#include "spmv/bandwidth.h"

// 获取 GPU 峰值带宽（缓存，std::call_once）
float peak = get_gpu_peak_bandwidth();

// 计算带宽利用率
float utilization = compute_bandwidth_utilization(bytes, time_ms);
```

## PageRank

```cpp
#include "spmv/pagerank.h"

PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

PageRankResult result = pagerank(adj, &config);

std::vector<TopKNode> top_k(10);
pagerank_top_k(&result, adj->num_rows, 10, top_k.data());

pagerank_free(&result);
```

## 错误处理

```cpp
#include "spmv/common.h"

// 语义化 CUDA 错误检查宏
CUDA_CHECK_MALLOC(cudaMalloc(...));
CUDA_CHECK_MEMCPY(cudaMemcpy(...));
CUDA_CHECK(cudaKernelCall);          // 向后兼容别名
```

| 错误码 | 含义 |
|--------|------|
| `SPMV_SUCCESS` | 成功 |
| `SPMV_ERROR_INVALID_INPUT` | 无效输入 |
| `SPMV_ERROR_CUDA_MALLOC` | GPU 内存分配失败 |
| `SPMV_ERROR_CUDA_MEMCPY` | 内存拷贝失败 |
| `SPMV_ERROR_CUDA_KERNEL` | Kernel 执行失败 |
