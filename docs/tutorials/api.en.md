---
layout: default
title: API Reference
nav_order: 4
has_children: false
lang: en
---

<p align="right">
  <a href="api">🇨🇳 简体中文</a>
</p>

# 📚 API Reference
{: .no_toc }

Complete GPU SpMV public API documentation.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Header Files Overview

| Header | Purpose |
|:-------|:-----|
| `<spmv/common.h>` | Error codes and basic definitions |
| `<spmv/cuda_buffer.h>` | RAII GPU memory management |
| `<spmv/csr_matrix.h>` | CSR sparse matrix |
| `<spmv/ell_matrix.h>` | ELL sparse matrix |
| `<spmv/spmv.h>` | SpMV computation interface |
| `<spmv/benchmark.h>` | Performance benchmarking framework |
| `<spmv/pagerank.h>` | PageRank algorithm |

---

## Error Handling

### SpMVError Enum

```cpp
enum class SpMVError {
    SUCCESS = 0,              // Operation successful
    INVALID_DIMENSION = -1,   // Matrix or vector dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failed
    CUDA_MEMCPY = -3,         // GPU memory copy failed
    KERNEL_LAUNCH = -4,       // CUDA kernel launch/execution failed
    INVALID_FORMAT = -5,      // Invalid sparse matrix format
    FILE_IO = -6,             // File read/write error
    OUT_OF_MEMORY = -7,       // Host/device out of memory
    INVALID_ARGUMENT = -8     // Invalid argument provided
};
```

### Error String

```cpp
const char* spmv_error_string(SpMVError err);
```

Returns a human-readable string describing the error.

**Example**:
```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
if (result.error != SpMVError::SUCCESS) {
    fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
}
```

---

## Data Structures

### CSRMatrix

CSR (Compressed Sparse Row) format sparse matrix.

```cpp
struct CSRMatrix {
    int num_rows;           // Number of rows
    int num_cols;           // Number of columns
    int nnz;                // Total non-zero elements

    float* values;          // Non-zero values array [nnz]
    int* col_indices;       // Column indices array [nnz]
    int* row_ptrs;          // Row pointers array [num_rows + 1]

    // GPU device pointers
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

**Invariants**:
- `row_ptrs[0] == 0`
- `row_ptrs[num_rows] == nnz`
- `row_ptrs[i] <= row_ptrs[i+1]` (for all i)

### CSRMatrix API

#### Create Matrix

```cpp
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
```

Create an empty CSR matrix structure.

**Parameters**:
- `num_rows`: Number of rows
- `num_cols`: Number of columns
- `nnz`: Number of non-zero elements

**Returns**: Newly allocated CSRMatrix pointer

#### Convert from Dense

```cpp
void csr_from_dense(CSRMatrix* csr, const float* dense, 
                    int num_rows, int num_cols);
```

Convert dense matrix to CSR format.

**Parameters**:
- `csr`: Target CSR matrix
- `dense`: Source dense matrix (row-major)
- `num_rows`: Number of rows
- `num_cols`: Number of columns

#### Transfer to GPU

```cpp
void csr_to_gpu(CSRMatrix* csr);
```

Transfer CSR matrix data to GPU.

**Parameters**:
- `csr`: CSR matrix to transfer

#### Destroy Matrix

```cpp
void csr_destroy(CSRMatrix* csr);
```

Free all memory used by CSR matrix.

**Parameters**:
- `csr`: CSR matrix to destroy

---

### ELLMatrix

ELLPACK format sparse matrix.

```cpp
struct ELLMatrix {
    int num_rows;           // Number of rows
    int num_cols;           // Number of columns
    int max_nnz_per_row;    // Maximum non-zero elements per row
    int nnz;                // Total non-zero elements

    float* values;          // Values array [num_rows * max_nnz_per_row]
    int* col_indices;       // Column indices array [num_rows * max_nnz_per_row]

    // GPU device pointers
    float* d_values;
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

### ELLMatrix API

#### Create Matrix

```cpp
ELLMatrix* ell_create(int num_rows, int num_cols, int max_nnz_per_row);
```

Create an empty ELL matrix structure.

#### Convert from CSR

```cpp
void ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
```

Convert CSR matrix to ELL format.

#### Transfer to GPU

```cpp
void ell_to_gpu(ELLMatrix* ell);
```

Transfer ELL matrix data to GPU.

#### Destroy Matrix

```cpp
void ell_destroy(ELLMatrix* ell);
```

Free all memory used by ELL matrix.

---

## SpMV Computation Interface

### SpMVConfig

SpMV computation configuration.

```cpp
struct SpMVConfig {
    KernelType kernel_type;  // Kernel type
    bool auto_select;        // Auto-select kernel
};
```

### KernelType Enum

```cpp
enum class KernelType {
    SCALAR_CSR,      // Scalar CSR kernel
    VECTOR_CSR,      // Vector CSR kernel
    MERGE_PATH,      // Merge Path kernel
    ELL              // ELL kernel
};
```

### Auto Configuration

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* csr);
```

Automatically select optimal kernel based on matrix characteristics.

**Parameters**:
- `csr`: CSR matrix

**Returns**: Optimized SpMVConfig

### Execute SpMV (CSR)

```cpp
SpMVResult spmv_csr(const CSRMatrix* csr, 
                    const float* d_x, 
                    float* d_y, 
                    const SpMVConfig* config, 
                    int n);
```

Execute CSR format SpMV computation: `y = A * x`

**Parameters**:
- `csr`: CSR matrix (must be on GPU)
- `d_x`: Input vector (GPU pointer)
- `d_y`: Output vector (GPU pointer)
- `config`: SpMV configuration
- `n`: Vector dimension

**Returns**: SpMVResult containing execution time and error code

### Execute SpMV (ELL)

```cpp
SpMVResult spmv_ell(const ELLMatrix* ell, 
                    const float* d_x, 
                    float* d_y, 
                    int n);
```

Execute ELL format SpMV computation: `y = A * x`

**Parameters**:
- `ell`: ELL matrix (must be on GPU)
- `d_x`: Input vector (GPU pointer)
- `d_y`: Output vector (GPU pointer)
- `n`: Vector dimension

**Returns**: SpMVResult containing execution time and error code

### SpMVResult

```cpp
struct SpMVResult {
    SpMVError error;      // Error code
    float time_ms;        // Execution time (milliseconds)
};
```

---

## RAII Memory Management

### CudaBuffer<T>

Template class for automatic GPU memory management.

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();
    
    T* data();
    const T* data() const;
    size_t size() const;
    
    // Copy disabled
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Move enabled
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
};
```

**Example**:
```cpp
{
    CudaBuffer<float> buffer(1000);
    // Access GPU pointer with buffer.data()
    // Automatically freed when leaving scope
}
```

---

## Performance Benchmarking

### BenchmarkConfig

```cpp
struct BenchmarkConfig {
    int iterations;           // Number of iterations
    bool warmup;              // Enable warmup
    bool print_details;       // Print detailed information
};
```

### Run Benchmark

```cpp
void spmv_benchmark(const CSRMatrix* csr, 
                    const BenchmarkConfig* config);
```

Run SpMV benchmark.

**Parameters**:
- `csr`: CSR matrix
- `config`: Benchmark configuration

---

## PageRank Algorithm

### PageRankConfig

```cpp
struct PageRankConfig {
    float damping;            // Damping factor (default 0.85)
    float tolerance;          // Convergence threshold (default 1e-6)
    int max_iterations;       // Maximum iterations (default 100)
};
```

### Execute PageRank

```cpp
SpMVResult spmv_pagerank(const CSRMatrix* csr, 
                         float* d_rank, 
                         const PageRankConfig* config);
```

Execute PageRank computation.

**Parameters**:
- `csr`: CSR matrix (adjacency matrix)
- `d_rank`: Rank vector (GPU pointer)
- `config`: PageRank configuration

**Returns**: SpMVResult containing convergence information

---

## Best Practices

### 1. Resource Management

Use RAII pattern for automatic resource management:

```cpp
// ✅ Recommended
void process() {
    CudaBuffer<float> x(1000);
    CudaBuffer<float> y(1000);
    // Automatic cleanup
}

// ❌ Avoid
void process() {
    float* x;
    cudaMalloc(&x, 1000 * sizeof(float));
    // Easy to forget cudaFree
}
```

### 2. Error Handling

Always check returned error codes:

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
if (result.error != SpMVError::SUCCESS) {
    // Handle error
    fprintf(stderr, "SpMV failed: %s\n", 
            spmv_error_string(result.error));
    return result.error;
}
```

### 3. Performance Optimization

Reuse execution context for best performance:

```cpp
SpMVExecutionContext ctx;
for (int i = 0; i < 100; i++) {
    // Texture objects created once, reused thereafter
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
}
```

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>Complete API specification see <a href="https://github.com/LessUp/gpu-spmv/blob/main/specs/api/public-api.md">specs/api/public-api.md</a></p>
</div>
