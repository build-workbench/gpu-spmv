# API Specification: GPU SpMV Library

> **Version**: v1.0.0  
> **Status**: ✅ Stable  
> **Last Updated**: 2025-04-16

---

## Overview

This document defines the public API specification for the GPU SpMV library. All implementations must adhere strictly to these interfaces.

---

## Header Files

| Header | Purpose |
|--------|---------|
| `<spmv/common.h>` | Error codes, CUDA helper macros |
| `<spmv/cuda_buffer.h>` | RAII GPU memory management |
| `<spmv/csr_matrix.h>` | CSR sparse matrix operations |
| `<spmv/ell_matrix.h>` | ELL sparse matrix operations |
| `<spmv/spmv.h>` | SpMV computation and kernel selection |
| `<spmv/bandwidth.h>` | Bandwidth metrics utilities |
| `<spmv/benchmark.h>` | Performance benchmarking framework |
| `<spmv/pagerank.h>` | PageRank algorithm interface |
| `<spmv/matrix_wrapper.h>` | Matrix format conversion utilities |
| `<spmv/test_utils.h>` | Testing utilities |

---

## Error Handling Specification

### Error Code Enum

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

### Error String Function

```cpp
const char* spmv_error_string(SpMVError err);
```

**Specification**:
- Returns a human-readable C-string describing the error
- Never returns NULL
- String is static and does not need to be freed

### CUDA Helper Macros

```cpp
#define CUDA_CHECK(call)           // General CUDA error check
#define CUDA_CHECK_MALLOC(call)    // Memory allocation check
#define CUDA_CHECK_MEMCPY(call)    // Memory copy check
```

---

## Data Structures

### CSRMatrix

**Purpose**: Compressed Sparse Row format for general-purpose sparse matrices.

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
- `row_ptrs[i] <= row_ptrs[i+1]` for all i
- All `col_indices[j]` must be in range `[0, num_cols)`

---

### ELLMatrix

**Purpose**: ELLPACK format for matrices with uniform row lengths.

```cpp
struct ELLMatrix {
    int num_rows;           // Number of rows
    int num_cols;           // Number of columns
    int max_nnz_per_row;    // Maximum non-zero elements per row
    int nnz;                // Actual total non-zero elements

    // Column-major storage for coalesced access
    float* values;          // Values array [num_rows * max_nnz_per_row]
    int* col_indices;       // Column indices [-1 indicates padding]

    float* d_values;        // GPU device pointers
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

**Invariants**:
- Padding elements use `col_indices == -1`
- Storage is column-major: `values[k * num_rows + i]` for row i, slot k

---

### SpMVConfig

**Purpose**: Configuration for SpMV kernel execution.

```cpp
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,     // One thread per row
        VECTOR_CSR,     // One warp (32 threads) per row
        MERGE_PATH,     // Perfect load balancing
        ELL_KERNEL      // ELL format专用 kernel
    };

    KernelType kernel_type;
    int block_size;         // CUDA block size (default: 256)
    bool use_texture;       // Enable texture cache for input vector
};
```

---

### SpMVResult

**Purpose**: Result metrics from SpMV computation.

```cpp
struct SpMVResult {
    float* y;               // Output vector (GPU pointer)
    float elapsed_ms;       // Execution time in milliseconds
    float gflops;           // Computational throughput
    float bandwidth_gb_s;   // Memory bandwidth utilization
    int error_code;         // 0 = success, negative = error
};
```

---

### PageRankConfig

**Purpose**: Configuration for PageRank algorithm.

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // Damping factor (d)
    float tolerance = 1e-6f;        // Convergence threshold
    int max_iterations = 100;       // Maximum iterations
};
```

---

### PageRankResult

**Purpose**: Results from PageRank computation.

```cpp
struct PageRankResult {
    float* ranks;           // PageRank scores [num_nodes]
    int iterations;         // Actual iterations performed
    float final_residual;   // Final L2 norm residual
    bool converged;         // Whether converged
    int error_code;         // Error code
};
```

---

### BenchmarkConfig

**Purpose**: Configuration for benchmarking.

```cpp
struct BenchmarkConfig {
    int iterations;         // Number of benchmark iterations
    bool compare_cpu;       // Whether to run CPU baseline
    bool export_json;       // Whether to export JSON report
    const char* json_path;  // Path for JSON output
};
```

---

### BenchmarkResult

**Purpose**: Results from benchmarking.

```cpp
struct BenchmarkResult {
    float avg_time_ms;      // Average execution time
    float min_time_ms;      // Minimum time
    float max_time_ms;      // Maximum time
    float stddev_ms;        // Standard deviation
    float gflops;           // GFLOPS achieved
    float bandwidth_gb_s;   // Bandwidth utilization
    float cpu_time_ms;      // CPU baseline time (if enabled)
};
```

---

## Core API Functions

### CSR Matrix Operations

```cpp
// Create empty CSR matrix
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);

// Destroy CSR matrix and free memory
void csr_destroy(CSRMatrix* matrix);

// Convert dense matrix to CSR format
int csr_from_dense(CSRMatrix* csr, const float* dense, 
                   int num_rows, int num_cols);

// Transfer CSR to GPU memory
int csr_to_gpu(CSRMatrix* csr);

// Transfer CSR from GPU to host memory
int csr_from_gpu(const CSRMatrix* csr);

// Get element at position (row, col)
float csr_get_element(const CSRMatrix* csr, int row, int col);

// Serialize CSR to binary file
int csr_serialize(const CSRMatrix* csr, const char* filename);

// Deserialize CSR from binary file
CSRMatrix* csr_deserialize(const char* filename);

// Compute CSR statistics
CSRStats csr_compute_stats(const CSRMatrix* csr);
```

**Return Values**: All functions returning `int` use `SpMVError` codes.

---

### ELL Matrix Operations

```cpp
// Create empty ELL matrix
ELLMatrix* ell_create(int num_rows, int num_cols, int max_nnz_per_row);

// Destroy ELL matrix and free memory
void ell_destroy(ELLMatrix* matrix);

// Convert dense matrix to ELL format
int ell_from_dense(ELLMatrix* ell, const float* dense,
                   int num_rows, int num_cols);

// Convert CSR to ELL format
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);

// Transfer ELL to GPU memory
int ell_to_gpu(ELLMatrix* ell);

// Transfer ELL from GPU to host memory
int ell_from_gpu(const ELLMatrix* ell);

// Serialize ELL to binary file
int ell_serialize(const ELLMatrix* ell, const char* filename);

// Deserialize ELL from binary file
ELLMatrix* ell_deserialize(const char* filename);
```

---

### SpMV Computation

```cpp
// Automatically select optimal kernel based on matrix characteristics
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// Execute SpMV on CSR format
SpMVResult spmv_csr(
    const CSRMatrix* A,           // Input matrix
    const float* d_x,             // Input vector (GPU)
    float* d_y,                   // Output vector (GPU)
    const SpMVConfig* config,     // Kernel configuration (optional)
    int vec_size,                 // Vector size (-1 for auto-detect)
    SpMVExecutionContext* context // Execution context for resource reuse
);

// Execute SpMV on ELL format
SpMVResult spmv_ell(
    const ELLMatrix* A,
    const float* d_x,
    float* d_y,
    const SpMVConfig* config,
    int vec_size,
    SpMVExecutionContext* context
);

// CPU reference implementation for validation
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);
```

**Preconditions**:
- Matrix must be valid (num_rows > 0, num_cols > 0)
- Input vector size must match matrix column count
- Output vector must be allocated with size matching matrix row count

**Postconditions**:
- `result.error_code == 0` on success
- Output vector `d_y` contains computed result
- Performance metrics populated in result struct

---

### PageRank Algorithm

```cpp
// Compute PageRank scores using iterative SpMV
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,  // Column-normalized adjacency matrix
    const PageRankConfig* config  // PageRank configuration
);

// Get top-K nodes by PageRank score
int pagerank_top_k(const PageRankResult* result, 
                   int num_nodes, 
                   int k, 
                   TopKNode* top_k);

// Free PageRank result memory
void pagerank_free(PageRankResult* result);
```

**Algorithm**: `r_{k+1} = d * A * r_k + (1-d) / n`

**Convergence**: When `||r_{k+1} - r_k||_2 < tolerance`

---

### Benchmarking Framework

```cpp
// Run CSR SpMV benchmark
BenchmarkResult benchmark_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config,
    const BenchmarkConfig* bench
);

// Export benchmark results to JSON
int benchmark_to_json(const BenchmarkResult* result, 
                      const char* filename);

// Import benchmark results from JSON
BenchmarkResult* benchmark_from_json(const char* filename);
```

---

### Bandwidth Metrics

```cpp
// Compute bandwidth for given operation
float compute_bandwidth(size_t bytes_accessed, float elapsed_ms);

// Get GPU peak bandwidth
float get_gpu_peak_bandwidth();
```

---

## Execution Context

### SpMVExecutionContext

**Purpose**: Reusable context for multiple SpMV operations (texture objects, etc.).

```cpp
struct SpMVExecutionContext {
    // Internal state for texture cache and resource reuse
    // Created on first use, destroyed on context destruction
};

// Initialize execution context
int spmv_context_init(SpMVExecutionContext* context);

// Destroy execution context and release resources
void spmv_context_destroy(SpMVExecutionContext* context);
```

---

## CudaBuffer<T> RAII Template

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();

    // Non-copyable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Movable
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    // Accessors
    T* get();
    const T* get() const;
    size_t size() const;

    // Memory operations
    void copyFromHost(const T* host_ptr, size_t count);
    void copyToHost(T* host_ptr, size_t count);
    void memset(int value);
    void fill(const T& value);
};
```

**Guarantees**:
- Constructor allocates GPU memory
- Destructor automatically frees GPU memory
- Exception-safe resource management

---

## Naming Conventions

| Category | Convention | Example |
|----------|------------|---------|
| Struct types | PascalCase | `CSRMatrix`, `SpMVConfig` |
| Functions | snake_case with prefix | `csr_create`, `spmv_csr` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_BLOCK_SIZE` |
| Enum values | UPPER_SNAKE_CASE | `SCALAR_CSR`, `VECTOR_CSR` |
| Private members | snake_case with underscore suffix | `ptr_`, `size_` |

---

## Versioning

This library follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

---

## Compatibility

| Component | Requirement |
|-----------|-------------|
| C++ Standard | C++17 or later |
| CUDA Toolkit | 11.0 or later (12.0+ recommended) |
| Compute Capability | 7.0+ (Volta) |
| Architecture Support | x86_64, ARM64 |
