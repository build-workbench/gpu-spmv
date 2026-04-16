---
layout: default
title: API Reference
lang: en
---

<p align="right">
  <a href="api.html">🇨🇳 简体中文</a>
</p>

# 📚 API Reference

This document provides complete API reference for GPU SpMV library, including all public interfaces, data structures, and error codes.

---

## Table of Contents

- [Header Overview](#header-overview)
- [Sparse Matrix Formats](#sparse-matrix-formats)
  - [CSR Format](#csr-compressed-sparse-row)
  - [ELL Format](#ell-ellpack)
- [SpMV Interface](#spmv-interface)
- [RAII Utilities](#raii-utilities)
- [Benchmarking](#benchmarking)
- [PageRank Algorithm](#pagerank-algorithm)
- [Error Handling](#error-handling)

---

## Header Overview

| Header | Description | Main Components |
|:-------|:------------|:----------------|
| `<spmv/common.h>` | Error codes & base definitions | `SpMVError`, `CUDA_CHECK_*` |
| `<spmv/cuda_buffer.h>` | RAII GPU memory management | `CudaBuffer<T>` |
| `<spmv/csr_matrix.h>` | CSR sparse matrix | `CSRMatrix`, `csr_*` functions |
| `<spmv/ell_matrix.h>` | ELL sparse matrix | `ELLMatrix`, `ell_*` functions |
| `<spmv/spmv.h>` | SpMV compute interface | `spmv_csr`, `spmv_ell` |
| `<spmv/bandwidth.h>` | Bandwidth metrics | `BandwidthMetrics` |
| `<spmv/benchmark.h>` | Performance testing | `benchmark_csr`, `benchmark_ell` |
| `<spmv/pagerank.h>` | PageRank algorithm | `pagerank`, `PageRankConfig` |

---

## Sparse Matrix Formats

### CSR (Compressed Sparse Row)

**Header**: `<spmv/csr_matrix.h>`

#### Data Structure

```cpp
struct CSRMatrix {
    int num_rows;          // Number of rows
    int num_cols;          // Number of columns
    int nnz;              // Total non-zero elements

    float* values;         // Non-zero values [nnz]
    int* col_indices;      // Column indices [nnz]
    int* row_ptrs;         // Row pointers [num_rows + 1]

    // GPU-side pointers
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

#### Core Functions

| Function | Signature | Description |
|:---------|:----------|:------------|
| `csr_create` | `CSRMatrix* csr_create(int rows, int cols, int nnz)` | Create CSR matrix |
| `csr_destroy` | `void csr_destroy(CSRMatrix* csr)` | Destroy CSR matrix |
| `csr_from_dense` | `int csr_from_dense(CSRMatrix* csr, const float* data, int rows, int cols)` | Convert from dense matrix |
| `csr_to_gpu` | `int csr_to_gpu(CSRMatrix* csr)` | Upload to GPU |
| `csr_from_gpu` | `int csr_from_gpu(CSRMatrix* csr)` | Download from GPU |
| `csr_compute_stats` | `CSRStats csr_compute_stats(const CSRMatrix* csr)` | Compute statistics |

**Statistics Structure**:
```cpp
struct CSRStats {
    float avg_nnz_per_row;   // Average non-zeros per row
    int max_nnz_per_row;     // Maximum non-zeros per row
    int min_nnz_per_row;     // Minimum non-zeros per row
    float skewness;          // Skewness = max / (min + 1)
};
```

---

### ELL (ELLPACK)

**Header**: `<spmv/ell_matrix.h>`

#### Data Structure

```cpp
struct ELLMatrix {
    int num_rows;
    int num_cols;
    int max_nnz_per_row;   // Max non-zeros per row
    int nnz;              // Actual non-zero count

    float* values;         // Column-major storage
    int* col_indices;      // Column indices (-1 for padding)

    float* d_values;
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

#### Core Functions

| Function | Signature | Description |
|:---------|:----------|:------------|
| `ell_create` | `ELLMatrix* ell_create(int rows, int cols, int max_nnz)` | Create ELL matrix |
| `ell_destroy` | `void ell_destroy(ELLMatrix* ell)` | Destroy ELL matrix |
| `ell_from_csr` | `int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr)` | CSR to ELL conversion |
| `ell_to_gpu` | `int ell_to_gpu(ELLMatrix* ell)` | Upload to GPU |

---

## SpMV Interface

**Header**: `<spmv/spmv.h>`

### Configuration Constants

```cpp
namespace spmv {
    constexpr int WARP_SIZE = 32;                    // Warp size
    constexpr int MIN_BLOCK_SIZE = 32;               // Minimum block
    constexpr int MAX_BLOCK_SIZE = 1024;             // Maximum block
    constexpr int DEFAULT_BLOCK_SIZE = 256;          // Default block
    constexpr int TEXTURE_CACHE_THRESHOLD_COLS = 10000;  // Texture cache threshold
}
```

### Configuration Structure

```cpp
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,   // One thread per row
        VECTOR_CSR,   // One warp per row
        MERGE_PATH,   // Even workload distribution
        ELL_KERNEL    // ELL format specific
    };

    KernelType kernel_type;
    int block_size = DEFAULT_BLOCK_SIZE;
    bool use_texture = false;
};
```

### Kernel Selection Strategy

| KernelType | Condition | Description |
|:-----------|:----------|:------------|
| `SCALAR_CSR` | `avg_nnz_per_row < 4` | Very sparse, minimal thread overhead |
| `VECTOR_CSR` | `skewness < 10` | Uniform distribution, warp collaborative |
| `MERGE_PATH` | `skewness >= 10` | Non-uniform, perfect load balancing |
| `ELL_KERNEL` | ELL format specific | Fully coalesced memory access |

### Core Functions

```cpp
// Automatically select optimal configuration
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* A,
                  const float* d_x,
                  float* d_y,
                  const SpMVConfig* config = nullptr,
                  int vec_size = -1,
                  SpMVExecutionContext* context = nullptr);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* A,
                  const float* d_x,
                  float* d_y,
                  const SpMVConfig* config = nullptr,
                  int vec_size = -1,
                  SpMVExecutionContext* context = nullptr);
```

### Result Structure

```cpp
struct SpMVResult {
    float* y;              // Output vector pointer
    float elapsed_ms;      // Execution time (ms)
    float gflops;          // Compute throughput
    float bandwidth_gb_s;  // Memory bandwidth
    int error_code;        // Error code (0=success)
};
```

### Execution Context

```cpp
// For reusing texture objects
class SpMVExecutionContext {
public:
    SpMVExecutionContext();
    ~SpMVExecutionContext();
    void reset();
};
```

**Example (multiple calls with reuse)**:
```cpp
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < 100; i++) {
    // Texture object created once, reused thereafter
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
```

---

## RAII Utilities

### CudaBuffer

**Header**: `<spmv/cuda_buffer.h>`

RAII template class for automatic GPU memory management:

```cpp
template<typename T>
class CudaBuffer {
public:
    // Construction/Destruction
    explicit CudaBuffer(size_t n);
    ~CudaBuffer();

    // Data transfer
    void copyFromHost(const T* host_ptr, size_t count);
    void copyToHost(T* host_ptr, size_t count) const;

    // Initialization
    void memset(int value);
    void fill(T value);

    // Access
    T* get();
    const T* get() const;
    size_t size() const;
    size_t bytes() const;
    bool empty() const;

    // Resize
    void resize(size_t new_size);
    void release();

    // Move semantics (copy disabled)
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
};
```

**Usage Example**:
```cpp
// Create buffers
CudaBuffer<float> d_x(1000);
CudaBuffer<float> d_y(1000);

// Initialize
d_x.memset(0);

// Host to device
d_x.copyFromHost(host_data, 1000);

// Device to host
d_y.copyToHost(result_data, 1000);
```

---

## Benchmarking

**Header**: `<spmv/benchmark.h>`

### Configuration Structure

```cpp
struct BenchmarkConfig {
    int num_warmup_runs = 5;    // Number of warmup runs
    int num_runs = 20;          // Number of test runs
    bool compare_cpu = true;    // Whether to compare with CPU
};
```

### Result Structures

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

struct ComparisonResult {
    BenchmarkResult gpu_result;
    BenchmarkResult cpu_result;
    float speedup;
    int error_code;
};
```

### Benchmark Functions

```cpp
// CSR benchmark
BenchmarkResult benchmark_csr(const CSRMatrix* A,
                             const float* x,
                             const SpMVConfig* config = nullptr,
                             const BenchmarkConfig* bench_config = nullptr);

// ELL benchmark
BenchmarkResult benchmark_ell(const ELLMatrix* A,
                             const float* x,
                             const BenchmarkConfig* bench_config = nullptr);

// GPU vs CPU comparison
ComparisonResult compare_gpu_cpu_csr(const CSRMatrix* A,
                                     const float* x,
                                     const SpMVConfig* config = nullptr,
                                     const BenchmarkConfig* bench_config = nullptr);
```

---

## PageRank Algorithm

**Header**: `<spmv/pagerank.h>`

### Configuration Structure

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // Damping factor
    float tolerance = 1e-6f;        // Convergence threshold
    int max_iterations = 100;       // Maximum iterations
};
```

### Result Structures

```cpp
struct PageRankResult {
    float* ranks;           // Rank scores array
    int iterations;         // Actual iterations
    float final_residual;   // Final residual
    bool converged;         // Whether converged
    int error_code;         // Error code
};

struct TopKNode {
    int node_id;
    float rank;
};
```

### Core Functions

```cpp
// Execute PageRank
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config = nullptr);

// Free results
void pagerank_free(PageRankResult* result);

// Get Top-K nodes
void pagerank_top_k(const PageRankResult* result,
                   int num_nodes,
                   int k,
                   TopKNode* top_k);
```

**Example**:
```cpp
// Create adjacency matrix and normalize
CSRMatrix* adj = /* ... */;
csr_to_gpu(adj);

// Configure
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

// Execute
PageRankResult result = pagerank(adj, &config);

if (result.converged) {
    printf("Converged after %d iterations\n", result.iterations);

    // Get Top-10
    std::vector<TopKNode> top10(10);
    pagerank_top_k(&result, adj->num_rows, 10, top10.data());
}

pagerank_free(&result);
csr_destroy(adj);
```

---

## Error Handling

**Header**: `<spmv/common.h>`

### Error Code Enumeration

```cpp
enum class SpMVError : int {
    SUCCESS = 0,              // Success
    INVALID_DIMENSION = -1,   // Dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failed
    CUDA_MEMCPY = -3,         // Memory copy failed
    KERNEL_LAUNCH = -4,       // Kernel launch/execution failed
    INVALID_FORMAT = -5,      // Matrix format error
    FILE_IO = -6,             // File I/O failed
    OUT_OF_MEMORY = -7,       // Out of memory
    INVALID_ARGUMENT = -8     // Invalid argument
};
```

### Error Handling Helpers

```cpp
// Get error description
const char* spmv_error_string(SpMVError err);

// Check macros (return error code)
#define CUDA_CHECK_MALLOC(call)
#define CUDA_CHECK_MEMCPY(call)

// Check macros (throw exception, for RAII classes)
#define CUDA_CHECK_THROW(call)
```

### Best Practices

```cpp
// Check returned error code
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);

if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    fprintf(stderr, "SpMV failed: %s\n",
            spmv_error_string(static_cast<SpMVError>(result.error_code)));
    // Error handling...
}
```

---

<div align="center">

**[← Architecture](architecture.en)** · **[ Examples →](examples.en)**

</div>
