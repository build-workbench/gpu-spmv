---
layout: default
title: API Reference
parent: Documentation
nav_order: 3
---

# 📚 API Reference
{: .no_toc }

Complete API documentation for GPU SpMV.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Headers Overview

| Header | Purpose |
|:-------|:--------|
| `<spmv/common.h>` | Error codes and definitions |
| `<spmv/cuda_buffer.h>` | RAII GPU memory |
| `<spmv/csr_matrix.h>` | CSR sparse matrix |
| `<spmv/ell_matrix.h>` | ELL sparse matrix |
| `<spmv/spmv.h>` | SpMV computation |
| `<spmv/benchmark.h>` | Performance testing |
| `<spmv/pagerank.h>` | PageRank algorithm |

---

## Core Data Structures

### CSRMatrix

```cpp
struct CSRMatrix {
    int num_rows, num_cols, nnz;
    float* values;
    int* col_indices;
    int* row_ptrs;
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

## Core Functions

### Auto Configuration

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

### SpMV Computation

```cpp
// CSR format
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x,
                    float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* ctx);

// ELL format
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x,
                    float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* ctx);
```

### Benchmarking

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

## Error Handling

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
