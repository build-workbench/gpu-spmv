---
layout: default
title: Architecture
nav_order: 6
permalink: /architecture.en
lang: en
---

<p align="right">
  <a href="architecture">🇨🇳 简体中文</a>
</p>

# Architecture
{: .no_toc }

System architecture, core algorithms, and design decisions.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│   PageRank  │  Iterative  │  Graph NNs  │  Scientific  │
├─────────────────────────────────────────────────────────┤
│                       API Layer                          │
│   spmv_csr  │  spmv_ell  │  benchmark  │  pagerank    │
├─────────────────────────────────────────────────────────┤
│                      Kernel Layer                        │
│  Scalar CSR │ Vector CSR │ Merge Path │  ELL Kernel   │
├─────────────────────────────────────────────────────────┤
│                      Storage Layer                       │
│              CSR Matrix      │      ELL Matrix         │
└─────────────────────────────────────────────────────────┘
```

**Design Principles**:
- **Layered Architecture**: Clear separation of storage, compute, and application layers
- **Strategy Pattern**: Pluggable kernel selection strategies
- **RAII Management**: Automatic resource lifecycle management

---

## Sparse Matrix Formats

### CSR (Compressed Sparse Row)

```
Matrix:            CSR Storage:
┌─────┬─────┐     values:      [1, 2, 3, 4, 5]
│ 1 0 2 │     col_indices: [0, 2, 1, 2, 3]
│ 0 3 4 │ =>  row_ptrs:    [0, 2, 4, 5]
│ 0 0 5 │     
└─────┴─────┘     row_ptrs[i] indicates row i's start
```

**Characteristics**:
- General format, efficient storage
- Suitable for irregular sparsity patterns
- Efficient row-wise traversal

### ELL (ELLPACK)

```
Matrix:            ELL Storage (column-major):
┌─────┬─────┐     values:      [1, 3, 5, 2, 4, 0]
│ 1 0 2 │     col_indices: [0, 1, 3, 2, 2, -]
│ 0 3 4 │     
│ 0 0 5 │     Padded to max row length for coalesced access
└─────┴─────┘
```

**Characteristics**:
- Column-major storage, coalesced access
- Suitable for uniform row lengths
- Best performance on GPU

---

## Kernel Selection Strategy

```
Input: CSR Matrix
       │
       ├── avg_nnz_per_row < 4 ──→ Scalar CSR
       │                            (1 thread/row, minimal overhead)
       │
       └── avg_nnz_per_row >= 4
               │
               ├── skewness < 10 ──→ Vector CSR
               │                      (Warp cooperation, balanced)
               │
               └── skewness >= 10 ──→ Merge Path
                                      (Perfect load balancing)
```

### Kernel Comparison

| Kernel | Threading | Best For | Bandwidth |
|:-------|:----------|:---------|:---------:|
| Scalar CSR | 1 thread/row | Very sparse | Medium |
| Vector CSR | 1 warp/row | Uniform dist | High |
| Merge Path | Dynamic | High skewness | Highest |
| ELL | Column-parallel | Uniform rows | Extreme |

---

## Design Decisions

### 1. RAII Resource Management

```cpp
// Automatic lifecycle management, prevents memory leaks
class CudaBuffer {
public:
    explicit CudaBuffer(size_t n) { cudaMalloc(&ptr_, n * sizeof(T)); }
    ~CudaBuffer() { cudaFree(ptr_); }
    // Disable copy, enable move
};
```

### 2. Execution Context

```cpp
// Cache texture objects to avoid recreation
SpMVExecutionContext ctx;
for (int i = 0; i < n_iter; i++) {
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
    // Texture objects are reused
}
```

### 3. Error Handling

```cpp
// Semantic error codes
enum class SpMVError {
    SUCCESS = 0,
    INVALID_DIMENSION = -1,
    CUDA_MALLOC = -2,
    // ...
};
```

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>More details in <a href="https://github.com/LessUp/gpu-spmv/tree/main/specs">specs/</a> directory</p>
</div>
