---
layout: default
title: API Reference
nav_order: 3
permalink: /api.en
lang: en
---

<p align="right">
  <a href="api">🇨🇳 简体中文</a>
</p>

# API Reference
{: .no_toc }

Complete GPU SpMV public API documentation.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Headers

```cpp
#include <spmv/spmv.h>       // Main interface
#include <spmv/csr_matrix.h> // CSR matrix
#include <spmv/ell_matrix.h> // ELL matrix
#include <spmv/cuda_buffer.h> // RAII memory management
#include <spmv/benchmark.h>  // Benchmarking
#include <spmv/pagerank.h>   // PageRank
```

---

## Error Handling

```cpp
enum class SpMVError {
    SUCCESS = 0,              // Success
    INVALID_DIMENSION = -1,   // Dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failed
    CUDA_MEMCPY = -3,         // Memory copy failed
    KERNEL_LAUNCH = -4,       // Kernel launch failed
    INVALID_FORMAT = -5,      // Invalid format
    FILE_IO = -6,             // File IO error
    OUT_OF_MEMORY = -7,       // Out of memory
    INVALID_ARGUMENT = -8     // Invalid argument
};

const char* spmv_error_string(SpMVError err);
```

---

## CSR Matrix

### Data Structure

```cpp
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    float* values;      // [nnz]
    int* col_indices;   // [nnz]
    int* row_ptrs;      // [num_rows + 1]
    float* d_values;    // GPU
    int* d_col_indices; // GPU
    int* d_row_ptrs;    // GPU
};
```

### API

```cpp
// Create/Destroy
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* csr);

// Data operations
void csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
void csr_to_gpu(CSRMatrix* csr);
void csr_from_gpu(CSRMatrix* csr);
```

---

## ELL Matrix

```cpp
struct ELLMatrix {
    int num_rows;
    int num_cols;
    int max_nnz_per_row;
    float* values;      // [num_rows * max_nnz_per_row]
    int* col_indices;   // [num_rows * max_nnz_per_row]
    float* d_values;
    int* d_col_indices;
};

// API
ELLMatrix* ell_create(int num_rows, int num_cols, int max_nnz_per_row);
void ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
void ell_to_gpu(ELLMatrix* ell);
void ell_destroy(ELLMatrix* ell);
```

---

## SpMV Computation

### Configuration

```cpp
enum class KernelType {
    SCALAR_CSR,    // 1 thread/row
    VECTOR_CSR,    // 1 warp/row
    MERGE_PATH,    // Load balanced
    ELL            // ELL format
};

struct SpMVConfig {
    KernelType kernel_type;
    bool auto_select;
};

struct SpMVResult {
    SpMVError error;
    float time_ms;
    float bandwidth_gbps;
    float bandwidth_utilization;
};
```

### Functions

```cpp
// Auto config
SpMVConfig spmv_auto_config(const CSRMatrix* csr);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* csr, 
                    const float* d_x, 
                    float* d_y, 
                    const SpMVConfig* config, 
                    int n);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* ell, 
                    const float* d_x, 
                    float* d_y, 
                    int n);
```

---

## RAII Memory Management

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();
    
    T* data();
    size_t size() const;
    
    // Disable copy, enable move
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
};
```

---

## PageRank

```cpp
struct PageRankConfig {
    float damping = 0.85f;
    float tolerance = 1e-6f;
    int max_iterations = 100;
};

SpMVResult pagerank(const CSRMatrix* csr, 
                    float* d_rank, 
                    const PageRankConfig* config);
```

---

## Complete Example

```cpp
#include <spmv/spmv.h>

int main() {
    // 1. Create CSR matrix
    CSRMatrix* csr = csr_create(1000, 1000, 10000);
    // ... fill data ...
    csr_to_gpu(csr);
    
    // 2. Prepare vectors
    CudaBuffer<float> d_x(1000), d_y(1000);
    
    // 3. Auto-config and execute
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 1000);
    
    // 4. Check result
    if (result.error != SpMVError::SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
        return 1;
    }
    
    printf("Bandwidth: %.1f%%\n", result.bandwidth_utilization * 100);
    
    csr_destroy(csr);
    return 0;
}
```

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>More examples on <a href="examples.en">Examples</a> page</p>
</div>
