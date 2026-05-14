# SpMV Kernels Design

## Context

Sparse Matrix-Vector Multiplication (SpMV) is a memory-bound operation on GPUs. The key challenge is handling irregular memory access patterns and load imbalance caused by variable row lengths in sparse matrices.

## Goals / Non-Goals

**Goals:**
- Maximize memory bandwidth utilization (>60% of theoretical peak)
- Handle matrices with any row length distribution efficiently
- Provide automatic kernel selection based on matrix characteristics

**Non-Goals:**
- Optimize for compute-bound operations
- Support multi-GPU SpMV
- Handle dense matrices (use cuBLAS instead)

## Decisions

### D1: Multiple Kernel Strategies

Four kernel types for different matrix patterns:

**Scalar CSR Kernel:**
```cpp
__global__ void spmv_csr_scalar(int num_rows, const int* row_ptrs,
    const int* col_indices, const float* values, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}
```
- Simple, no synchronization overhead
- Best for: Very sparse matrices (avg_nnz < 4)

**Vector CSR Kernel:**
```cpp
__global__ void spmv_csr_vector(int num_rows, const int* row_ptrs,
    const int* col_indices, const float* values, const float* x, float* y) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id < num_rows) {
        float sum = 0.0f;
        for (int j = row_ptrs[warp_id] + lane_id;
             j < row_ptrs[warp_id + 1]; j += 32) {
            sum += values[j] * x[col_indices[j]];
        }

        // Warp-level reduction using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) y[warp_id] = sum;
    }
}
```
- Efficient warp-level reduction
- Best for: Uniform row lengths (skewness < 10)

**Merge Path Kernel:**
- Treats row pointer and non-zero sequences as ordered paths
- Uses binary search to find uniform split points
- Best for: Highly skewed matrices (skewness >= 10)

### D2: Kernel Selection Heuristic

Selection logic is extracted into a pure function `select_kernel(CSRStats, int, SpMVThresholds)` in the internal `kernel_selector` module, making it independently testable and free of global state.

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* A) {
    if (!A || A->num_rows < 0) {
        return SpMVConfig(SpMVConfig::SCALAR_CSR, DEFAULT_BLOCK_SIZE, false);
    }
    CSRStats stats = csr_compute_stats(A);
    return select_kernel(stats, A->num_cols, spmv_get_thresholds());
}
```

**Rationale**: Simple heuristic based on empirical performance analysis. Pure-function extraction improves testability and eliminates hidden global dependencies.

### D3: Texture Cache for Input Vector

```cpp
// Use SpMVExecutionContext to reuse texture objects
SpMVExecutionContext context;
config.use_texture = true;

for (int i = 0; i < iterations; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
```

`SpMVExecutionContext` is implemented as a class with encapsulated CUDA texture state (not a public struct). Clients interact only through `reset()` and `is_texture_bound()`.

**Rationale**: Texture cache provides cached access to input vector x, beneficial when x is accessed multiple times (irregular pattern) or when matrix fits in L2 cache. Hiding CUDA primitives prevents accidental direct manipulation of texture objects.

### D4: Warp-Level Reduction

Using shuffle instructions instead of shared memory:
```cpp
// No bank conflicts, fully parallel
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```

**Rationale**: Shuffle instructions are faster and avoid shared memory bank conflicts.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Kernel selection may be suboptimal for edge cases | Allow manual override via SpMVConfig |
| Merge Path has higher overhead | Only use when skewness indicates benefit |
| Texture cache adds complexity | Make it optional via use_texture flag |

## Performance Targets

| Metric | Target |
|--------|--------|
| Bandwidth Utilization | > 60% of theoretical peak |
| GFLOPS | Proportional to bandwidth (2 ops per element) |
| Load Balance Efficiency | > 70% for skewed matrices |
