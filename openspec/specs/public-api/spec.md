# Public API Specification

> **Version**: v1.0.0
> **Status**: ✅ Stable
> **Last Updated**: 2025-04-16

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

## Requirement: Error Handling API
**Name**: error-api
**Text**: Provide consistent error handling across all API functions.

### Scenario: Error Code Return
**WHEN** any API function encounters an error
**THEN** should return appropriate SpMVError enum value

### Scenario: Error String Conversion
**WHEN** calling spmv_error_string with an error code
**THEN** should return human-readable C-string description

---

## Requirement: CSR Matrix API
**Name**: csr-api
**Text**: Provide API for CSR matrix operations.

### Scenario: Matrix Creation
**WHEN** calling csr_create with valid dimensions
**THEN** should return allocated CSRMatrix pointer

### Scenario: Dense Conversion
**WHEN** calling csr_from_dense with a dense matrix
**THEN** should convert to CSR format preserving all non-zero elements

### Scenario: GPU Transfer
**WHEN** calling csr_to_gpu with a valid CSR matrix
**THEN** should allocate and copy data to GPU memory

---

## API Functions

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
int benchmark_to_json(const BenchmarkResult* result, const char* filename);

// Import benchmark results from JSON
BenchmarkResult* benchmark_from_json(const char* filename);
```

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

## See Also

- [CSR Format](../csr-format/spec.md) - CSR format details
- [ELL Format](../ell-format/spec.md) - ELL format details
- [SpMV Kernels](../spmv-kernels/spec.md) - Kernel implementations
