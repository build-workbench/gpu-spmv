---
layout: default
title: API 参考
parent: 中文文档
nav_order: 3
lang: zh
---

# 📚 API 参考
{: .no_toc }

完整的 API 接口文档。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 头文件概览

| 头文件 | 功能 |
|:-------|:-----|
| `<spmv/common.h>` | 错误码与基础定义 |
| `<spmv/cuda_buffer.h>` | RAII GPU 内存管理 |
| `<spmv/csr_matrix.h>` | CSR 稀疏矩阵 |
| `<spmv/ell_matrix.h>` | ELL 稀疏矩阵 |
| `<spmv/spmv.h>` | SpMV 计算接口 |
| `<spmv/benchmark.h>` | 性能测试框架 |
| `<spmv/pagerank.h>` | PageRank 算法 |

---

## 核心数据结构

### CSRMatrix

```cpp
struct CSRMatrix {
    int num_rows, num_cols, nnz;
    float* values;         // [nnz]
    int* col_indices;      // [nnz]
    int* row_ptrs;         // [num_rows + 1]
    // GPU 指针...
};
```

### SpMVConfig

```cpp
struct SpMVConfig {
    enum KernelType { SCALAR_CSR, VECTOR_CSR, 
                      MERGE_PATH, ELL_KERNEL };
    KernelType kernel_type;
    int block_size = DEFAULT_BLOCK_SIZE;
    bool use_texture = false;
};
```

### SpMVResult

```cpp
struct SpMVResult {
    float* y;
    float elapsed_ms;
    float gflops;
    float bandwidth_gb_s;
    int error_code;
};
```

---

## 核心函数

### 自动配置

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

### SpMV 计算

```cpp
// CSR 格式
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, 
                    float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* ctx);

// ELL 格式
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x,
                    float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* ctx);
```

### 基准测试

```cpp
BenchmarkResult benchmark_csr(const CSRMatrix* A, 
                              const float* x,
                              const SpMVConfig* config,
                              const BenchmarkConfig* bench);
```

### PageRank

```cpp
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config);
```

---

## 错误处理

```cpp
enum class SpMVError {
    SUCCESS = 0,
    INVALID_DIMENSION = -1,
    CUDA_MALLOC = -2,
    CUDA_MEMCPY = -3,
    KERNEL_LAUNCH = -4,
    INVALID_FORMAT = -5,
    FILE_IO = -6,
    OUT_OF_MEMORY = -7,
    INVALID_ARGUMENT = -8
};

const char* spmv_error_string(SpMVError err);
```
