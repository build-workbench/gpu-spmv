---
layout: default
title: Architecture
lang: en
---

<p align="right">
  <a href="architecture.html">🇨🇳 简体中文</a>
</p>

# 🏗️ Architecture Design

This document details the architectural design, core algorithms, and design decisions of the GPU SpMV library.

---

## Table of Contents

- [System Overview](#system-overview)
- [Sparse Matrix Formats](#sparse-matrix-formats)
- [Kernel Design](#kernel-design)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Design Decisions](#design-decisions)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │
│  │  PageRank   │  │   Solvers   │  │   Other SpMV Apps       │   │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘   │
├───────┬─┴────────────────┴───────────────────┬─┴─────────────────┤
│       │           Interface Layer            │                   │
│       ▼                                      ▼                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                      SpMV Runtime API                     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │ spmv_csr │ │ spmv_ell │ │ benchmark│ │   pagerank   │  │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │    │
│  └───────┼────────────┼────────────┼──────────────┼──────────┘    │
├──────────┼────────────┼────────────┴──────────────┼────────────────┤
│          ▼            ▼                           ▼                │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │                     Kernel Scheduler Layer                 │     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │     │
│  │  │ Scalar   │ │ Vector   │ │  Merge   │ │     ELL      │  │     │
│  │  │   CSR    │ │   CSR    │ │   Path   │ │   Kernel     │  │     │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │     │
│  └──────────────────────────────────────────────────────────┘     │
├───────────────────────────────────────────────────────────────────┤
│                        Storage Layer                              │
│  ┌────────────────────┐    ┌────────────────────┐                 │
│  │    CSR Matrix      │    │    ELL Matrix      │                 │
│  │  (values, indices, │    │  (values, indices, │                 │
│  │   row_ptrs)        │    │   max_nnz_per_row) │                 │
│  └────────────────────┘    └────────────────────┘                 │
├───────────────────────────────────────────────────────────────────┤
│                        Runtime Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │
│  │ CudaBuffer   │  │ CudaTimer    │  │ SpMVExecutionContext │     │
│  │ (RAII Mem)   │  │ (Events)     │  │ (Texture Cache)      │     │
│  └──────────────┘  └──────────────┘  └──────────────────────┘     │
├───────────────────────────────────────────────────────────────────┤
│                      CUDA Driver / Runtime                        │
└───────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Responsibility | Key Components |
|:------|:---------------|:---------------|
| **Application** | End-user applications | PageRank, Iterative solvers |
| **Interface** | Provide user APIs | `spmv_csr`, `benchmark_csr` |
| **Scheduler** | Kernel selection & execution | Auto-config, kernel dispatch |
| **Storage** | Matrix data management | CSR/ELL data structures |
| **Runtime** | Resource & context management | RAII classes, execution context |

---

## Sparse Matrix Formats

### CSR (Compressed Sparse Row)

CSR format compresses sparse matrix storage using three arrays:

```
Original Matrix (3×4):
┌─────┬─────┬─────┬─────┐
│  1  │  0  │  2  │  0  │  Row 0
├─────┼─────┼─────┼─────┤
│  0  │  3  │  0  │  4  │  Row 1
├─────┼─────┼─────┼─────┤
│  5  │  0  │  0  │  6  │  Row 2
└─────┴─────┴─────┴─────┘

CSR Representation:
┌──────────────────────────────────────────────┐
│ values:      [1, 2, 3, 4, 5, 6]             │
│ col_indices: [0, 2, 1, 3, 0, 3] ← col index │
│ row_ptrs:    [0, 2, 4, 6]       ← row start │
└──────────────────────────────────────────────┘

Access row i: data range [row_ptrs[i], row_ptrs[i+1])
```

**Complexity Analysis:**
- Storage: `O(nnz + rows)` — independent of matrix size
- Random access: `O(log nnz_per_row)` — requires binary search
- Row traversal: `O(nnz_per_row)` — sequential memory access

### ELL (ELLPACK)

ELL format uses fixed-width 2D arrays stored in column-major order:

```
Original Matrix (4×4, max_nnz_per_row=2):
┌─────┬─────┬─────┬─────┐
│  1  │  0  │  2  │  0  │  Row 0: [1, 2]
├─────┼─────┼─────┼─────┤
│  0  │  3  │  0  │  4  │  Row 1: [3, 4]
├─────┼─────┼─────┼─────┤
│  5  │  6  │  0  │  0  │  Row 2: [5, 6]
├─────┼─────┼─────┼─────┤
│  7  │  0  │  0  │  0  │  Row 3: [7, -] ← padding
└─────┴─────┴─────┴─────┘

ELL Column-major Storage:
┌─────────────────────────────────────────┐
│ values:      [1, 3, 5, 7, 2, 4, 6, 0]  │
│                ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑   │
│               row0 row1 row2 row3   (k=0)
│ col_indices: [0, 1, 0, 0, 2, 3, 1, -1] │
│                ↑  ↑  ↑  ↑              │
│               (k=1)                    │
│ Layout: [row0_k0, row1_k0, ..., row0_k1, ...] │
└─────────────────────────────────────────┘
```

**GPU Access Optimization:**
- Thread i accesses column k: `index = k * num_rows + i`
- Adjacent threads access consecutive addresses → fully coalesced
- No bank conflicts

---

## Kernel Design

### Scheduling Strategy Decision Tree

```
                 Matrix Feature Analysis
                          │
         ┌────────────────┴────────────────┐
         │                                 │
    ┌────┴────┐                    ┌───────┴────────┐
    │ FORMAT  │                    │   Statistics   │
    └────┬────┘                    └───────┬────────┘
         │                                 │
    ELL ─┼─→ ELL Kernel            ┌───────┴───────┐
         │                         │               │
        CSR                  avg_nnz < 4?   skewness < 10?
                                    │             │
                                  Yes            No
                                    │             │
                             ┌──────┴──────┐   ┌───┴────┐
                             │   Scalar    │   │ Vector │──Yes──→ Vector CSR
                             │     CSR     │   │  CSR   │
                             └─────────────┘   └────────┘
                                                      │
                                                     No
                                                      │
                                                ┌─────┴─────┐
                                                │   Merge   │
                                                │   Path    │
                                                └───────────┘
```

### Kernel Characteristics

| Kernel | Parallel Strategy | Sync Overhead | Load Balancing | Best For |
|:-------|:------------------|:-------------:|:--------------:|:---------|
| **Scalar** | 1 thread/row | None | Poor (row length variance) | avg_nnz < 4 |
| **Vector** | 1 warp/row | Warp shuffle | Medium | Uniform distribution |
| **Merge Path** | Equal partitioning | Atomic ops | Perfect | Highly irregular |
| **ELL** | 1 thread/row | None | - | ELL format matrices |

### Merge Path Algorithm

Merge Path solves CSR load imbalance issues:

```
Row pointers:   [0, 100, 101, 200]    (4 rows, highly uneven)
Non-zeros:      [0, ..., 99, 100, 101, ..., 199]

Traditional CSR Problem:
- Row 0: 100 non-zeros → 1 thread processes 100 elements
- Row 1: 1 non-zero   → 1 thread processes 1 element
→ Severe load imbalance!

Merge Path Solution:
1. Treat row pointers and non-zeros as two sorted sequences
2. Partition work uniformly along the merge path
3. Each thread block processes equal (row, offset) ranges

Diagonal cut:
         row:  0      1    2      3      4
              [0]─[100]─[101]─[200]
               │╲   │╲    │╲    │╲
              [0]..[99][100][101]..[199]
               nz:

Partition: 4 thread blocks, each handles ~50 non-zeros
→ Perfect load balance!
```

---

## Memory Management

### RAII Design Pattern

All GPU resources use RAII (Resource Acquisition Is Initialization):

```cpp
// CudaBuffer: automatic GPU memory management
template<typename T>
class CudaBuffer {
    T* d_ptr_;           // Device pointer
    size_t size_;        // Element count
    size_t capacity_;    // Allocated capacity

public:
    explicit CudaBuffer(size_t n);   // Constructor allocates
    ~CudaBuffer();                    // Destructor frees
    
    // Disable copy, allow move
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept;
};

// Usage: exception-safe
try {
    CudaBuffer<float> buf(1000000);  // Allocate
    process_data(buf.get());          // Use
    // Even if exception thrown, buf is properly freed
} catch (...) {
    // No manual cleanup needed
}
```

### Execution Context Reuse

`SpMVExecutionContext` reuses expensive texture objects:

```cpp
class SpMVExecutionContext {
    cudaTextureObject_t tex_x_;     // Texture object (expensive to create)
    const float* cached_x_;         // Cached vector pointer
    size_t cached_x_length_;        // Cached vector length

public:
    SpMVExecutionContext();         // Constructor
    ~SpMVExecutionContext();        // Destructor destroys texture

    cudaTextureObject_t prepare_texture(
        const float* x, size_t n);  // Lazy create/reuse
};

// Performance: reuse texture across multiple calls
SpMVExecutionContext ctx;
for (int iter = 0; iter < 100; iter++) {
    // Texture created once, reused thereafter
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
}
```

---

## Error Handling

### Layered Error Handling

```
┌─────────────────────────────────────────┐
│           User Layer                    │
│    if (result.error_code != 0) {...}    │
├─────────────────────────────────────────┤
│           API Layer                     │
│    return SpMVResult{..., error_code};  │
├─────────────────────────────────────────┤
│           Runtime Layer                 │
│    CUDA_CHECK_MALLOC(cudaMalloc(...));  │
├─────────────────────────────────────────┤
│           CUDA Driver Layer             │
│    cudaError_t err = cudaMalloc(...);   │
└─────────────────────────────────────────┘
```

### Error Code System

```cpp
enum class SpMVError : int {
    SUCCESS = 0,              // Success
    INVALID_DIMENSION = -1,   // Dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failed
    CUDA_MEMCPY = -3,         // Memory copy failed
    KERNEL_LAUNCH = -4,       // Kernel launch failed
    INVALID_FORMAT = -5,      // Matrix format error
    FILE_IO = -6,             // File operation failed
    OUT_OF_MEMORY = -7,       // Out of memory
    INVALID_ARGUMENT = -8     // Invalid argument
};

// Error message conversion
const char* spmv_error_string(SpMVError err);
// Returns: "Invalid dimension", "CUDA malloc failed", ...
```

---

## Performance Optimization

### Memory Bandwidth Optimization

SpMV is **memory-bandwidth-bound**. The focus is maximizing bandwidth utilization:

```
Arithmetic Intensity = Computation / Data Movement
                     = (2 × nnz FLOP) / (~20 × nnz bytes)
                     ≈ 0.1 FLOP/Byte  <<  GPU requirement (~10 FLOP/Byte)

→ Must optimize memory access patterns
```

**Optimization Techniques:**

| Technique | Implementation | Effect |
|:----------|:---------------|:-------|
| **Coalesced Access** | ELL Column-major | Eliminates non-coalesced penalty |
| **Texture Cache** | `cudaTextureObject_t` | Improves random access performance |
| **Warp Shuffle** | `__shfl_down_sync` | Eliminates bank conflicts |
| **Prefetching** | Compiler auto-optimization | Hides memory latency |

### Load Balancing

```
Matrix Skewness = max_nnz_per_row / (min_nnz_per_row + 1)

┌───────────────────────┬────────────────┬────────────────┐
│ Skewness              │ Best Kernel    │ Reason         │
├───────────────────────┼────────────────┼────────────────┤
│ < 10 (Uniform)        │ Vector CSR     │ Efficient warp │
│ >= 10 (Non-uniform)   │ Merge Path     │ Perfect balance│
│ < 4 (Very sparse)     │ Scalar CSR     │ Min overhead   │
└───────────────────────┴────────────────┴────────────────┘
```

---

## Design Decisions

### ADR-001: Static vs Dynamic Libraries

**Decision**: Provide static library (`.a`) instead of dynamic library (`.so`/`.dll`)

**Rationale**:
- ✅ Simplified deployment, no runtime dependencies
- ✅ Link-time optimization opportunities
- ✅ More stable CUDA runtime static linking
- ✅ Avoid symbol versioning issues

**Trade-offs**:
- ❌ Larger executable size
- ❌ Library updates require recompilation

### ADR-002: RAII Resource Management

**Decision**: Use RAII pattern for all GPU resources

**Rationale**:
- ✅ Exception-safe
- ✅ Prevents memory leaks
- ✅ Cleaner code

**Implementation**:
- `CudaBuffer<T>`: GPU memory
- `CudaTimer`: CUDA Events
- `SpMVExecutionContext`: Texture objects

### ADR-003: Automatic Kernel Selection

**Decision**: Automatically select optimal kernel based on matrix statistics

**Criteria**:
```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* csr) {
    auto stats = csr_compute_stats(csr);
    
    if (stats.avg_nnz_per_row < 4)
        return {SCALAR_CSR, ...};
    if (stats.skewness < 10)
        return {VECTOR_CSR, ...};
    return {MERGE_PATH, ...};
}
```

**Benefits**:
- Users don't need kernel details
- Automatically adapts to different matrix features
- Guarantees good performance baseline

### ADR-004: Texture Cache Reuse

**Decision**: Provide `SpMVExecutionContext` for texture object reuse

**Background**:
- Creating/destroying `cudaTextureObject_t` is expensive (~100μs)
- Iterative algorithms (PageRank) need frequent SpMV calls

**Solution**:
- Execution context caches texture objects
- Lazy creation, automatic reuse
- Automatic cleanup on destruction

---

<div align="center">

**[← Installation](installation.en)** · **[ API Reference →](api.en)**

</div>
