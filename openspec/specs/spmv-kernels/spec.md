# SpMV Kernels

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: SpMV CUDA Kernels
**Name**: spmv-cuda-kernels
**Text**: Provide multiple optimized CUDA kernels for sparse matrix-vector multiplication with automatic kernel selection.

### Scenario: CSR SpMV Correctness
**WHEN** executing SpMV with CSR format
**THEN** should correctly compute y = A * x with relative error < 1e-6 compared to CPU reference

### Scenario: ELL SpMV Correctness
**WHEN** executing SpMV with ELL format
**THEN** should correctly compute y = A * x with relative error < 1e-6 compared to CPU reference

### Scenario: Empty Row Handling
**WHEN** processing matrices where some rows have zero non-zero elements
**THEN** SpMV should handle correctly and produce 0 for those rows

### Scenario: Dimension Validation
**WHEN** input vector dimensions don't match matrix column count
**THEN** should return INVALID_DIMENSION error code

### Scenario: Bandwidth Utilization
**WHEN** executing optimized SpMV
**THEN** should achieve at least 60% of theoretical peak memory bandwidth

---

## Requirement: Load Balancing
**Name**: spmv-load-balancing
**Text**: Provide load-balanced SpMV kernels to avoid performance degradation due to uneven row lengths.

### Scenario: Vector CSR Kernel
**WHEN** processing rows with different lengths using Vector CSR kernel
**THEN** one warp (32 threads) should be allocated per row with threads cooperating on non-zero elements

### Scenario: Merge Path Load Balancing
**WHEN** matrix row lengths are highly skewed (max/min > 100)
**THEN** Merge Path kernel should distribute work evenly and maintain at least 70% efficiency

### Scenario: Kernel Selection
**WHEN** calling spmv_auto_config
**THEN** should select appropriate kernel based on matrix characteristics:
- avg_nnz_per_row < 4 → SCALAR_CSR
- avg_nnz_per_row >= 4 AND skewness < 10 → VECTOR_CSR
- avg_nnz_per_row >= 4 AND skewness >= 10 → MERGE_PATH

---

## Requirement: Bandwidth Optimization
**Name**: spmv-bandwidth-optimization
**Text**: Maximize GPU memory throughput for bandwidth-bound SpMV operations.

### Scenario: Coalesced Access
**WHEN** accessing matrix data
**THEN** should use coalesced memory access patterns where possible

### Scenario: Texture Cache
**WHEN** texture memory caching is enabled for input vector x
**THEN** should improve cache hit rate for repeated access patterns

### Scenario: Bandwidth Metrics
**WHEN** SpMV operation completes
**THEN** should provide bandwidth utilization metrics in result structure

---

## Kernel Types

| Kernel | Strategy | Best For |
|--------|----------|----------|
| Scalar CSR | 1 thread per row | Very sparse (avg_nnz < 4) |
| Vector CSR | 1 warp per row | Uniform distribution (skewness < 10) |
| Merge Path | Load-balanced partitioning | Skewed matrices (skewness >= 10) |
| ELL Kernel | Column-major access | Uniform row lengths |

## Kernel Selection Flow

```
Matrix Feature Analysis
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

## Data Structures

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

struct SpMVResult {
    float* y;               // Output vector (GPU pointer)
    float elapsed_ms;       // Execution time in milliseconds
    float gflops;           // Computational throughput
    float bandwidth_gb_s;   // Memory bandwidth utilization
    int error_code;         // 0 = success, negative = error
};
```

## Test Properties

| Property | Description |
|----------|-------------|
| P8 | SpMV CSR Correctness |
| P9 | SpMV ELL Correctness |
| P10 | SpMV Dimension Validation |
| P11 | Kernel Selector Validity |
| P12 | Bandwidth Metrics Validity |

## See Also

- [CSR Format](../csr-format/spec.md) - CSR matrix format
- [ELL Format](../ell-format/spec.md) - ELL matrix format
- [Public API](../public-api/spec.md) - SpMV API functions
