# Product Requirements: GPU SpMV (Sparse Matrix-Vector Multiplication)

> **Version**: v1.0.0  
> **Status**: ✅ Implemented  
> **Last Updated**: 2025-04-16

---

## Overview

This project implements a GPU-based Sparse Matrix-Vector Multiplication (SpMV) operator, supporting CSR and ELL format sparse matrix storage. The project addresses key challenges when GPUs handle sparse data:

1. **Irregular Memory Access** — Optimized via Column-major storage and texture cache
2. **Load Imbalance** — Addressed through multiple kernel strategies and auto-selection
3. **Bandwidth Utilization Optimization** — Improved via coalesced access and Warp-level reduction

---

## Glossary

| Term | Definition |
|------|------------|
| SpMV | Sparse Matrix-Vector Multiplication |
| CSR | Compressed Sparse Row format, using three arrays to store sparse matrices |
| ELL | ELLPACK format, padding each row to the same length for sparse matrix storage |
| Warp | A group of 32 threads that execute together in GPU |
| Warp Divergence | Performance loss when threads within the same warp execute different branches |
| Load Balancing | Ensuring workload is evenly distributed across all threads |
| Bandwidth Bound | Performance limited by memory bandwidth rather than compute capability |
| Coalesced Access | Adjacent threads accessing consecutive memory addresses to maximize bandwidth |
| PageRank | Web ranking algorithm based on iterative matrix-vector multiplication |

---

## Functional Requirements

### REQ-1: CSR Format Sparse Matrix Storage

**User Story**: As a developer, I want to store sparse matrices in CSR format so that I can efficiently represent matrices with大量 zero elements using minimal memory.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 1.1 | CSR storage should use three arrays: values (non-zero element values), column_indices (column indices), row_pointers (row pointers) |
| 1.2 | When converting dense matrix to CSR format, all non-zero elements and their positions should be preserved accurately |
| 1.3 | When querying element at position (i, j), should return correct value (non-zero or zero) |
| 1.4 | Support matrices with up to 10 million non-zero elements |
| 1.5 | Binary serialization format should deserialize to equivalent CSR structure |

**Implementation Status**: ✅ Complete

---

### REQ-2: ELL Format Sparse Matrix Storage

**User Story**: As a developer, I want to store sparse matrices in ELL format so that I can achieve better GPU memory coalesced access for matrices with uniform row lengths.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 2.1 | ELL storage should use two 2D arrays: values and column_indices, with each row padded to max_nnz_per_row |
| 2.2 | When converting dense matrix to ELL format, all non-zero elements and their positions should be preserved accurately |
| 2.3 | When a row has fewer non-zero elements than max_nnz_per_row, should pad with zeros and invalid column indices (-1) |
| 2.4 | Data should be stored in column-major order for GPU coalesced access |
| 2.5 | Binary serialization format should deserialize to equivalent ELL structure |

**Implementation Status**: ✅ Complete

---

### REQ-3: Basic SpMV CUDA Kernel

**User Story**: As a developer, I want a basic SpMV CUDA kernel so that I can perform sparse matrix-vector multiplication on the GPU.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 3.1 | When executing SpMV with CSR format, should correctly compute y = A * x |
| 3.2 | When executing SpMV with ELL format, should correctly compute y = A * x |
| 3.3 | GPU results vs CPU reference implementation relative error should be within 1e-6 (single precision) |
| 3.4 | Should handle matrices where some rows have zero non-zero elements |
| 3.5 | Should return error code when input vector dimensions don't match matrix column count |

**Implementation Status**: ✅ Complete

---

### REQ-4: Load Balancing Optimization

**User Story**: As a developer, I want load-balanced SpMV kernels to avoid performance degradation due to uneven row lengths.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 4.1 | Vector CSR kernel should allocate one warp (32 threads) per row, with threads cooperating on non-zero elements |
| 4.2 | Vector CSR should achieve at least 80% of peak performance for uniform-length matrices when processing rows of different lengths |
| 4.3 | Merge Path kernel should distribute work evenly across all threads regardless of row length distribution |
| 4.4 | When matrix row lengths are highly skewed (max/min > 100), Merge Path should maintain at least 70% efficiency |
| 4.5 | Should provide kernel selection function based on matrix characteristics |

**Implementation Status**: ✅ Complete

---

### REQ-5: Bandwidth Optimization

**User Story**: As a developer, I want bandwidth-optimized SpMV implementations to maximize GPU memory throughput for bandwidth-bound operations.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 5.1 | Optimized SpMV should achieve at least 60% of theoretical peak memory bandwidth on target GPU |
| 5.2 | Should use coalesced memory access patterns when accessing matrix data where possible |
| 5.3 | Should support texture memory caching for input vector x to improve cache hit rate |
| 5.4 | Should demonstrate cache reuse performance improvement when matrix fits in L2 cache |
| 5.5 | Should provide bandwidth utilization metrics after each SpMV operation |

**Implementation Status**: ✅ Complete

---

### REQ-6: Performance Benchmarking

**User Story**: As a developer, I want comprehensive benchmarking tools to measure and compare SpMV performance across different implementations.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 6.1 | Benchmark suite should measure execution time, GFLOPS, and bandwidth utilization for each SpMV kernel |
| 6.2 | Should support standard sparse matrix test sets (e.g., SuiteSparse collection) |
| 6.3 | Should report avg, min, max, and stddev across multiple runs |
| 6.4 | Should compare GPU implementation against CPU baseline |
| 6.5 | Should generate JSON-formatted performance reports for analysis |

**Implementation Status**: ✅ Complete

---

### REQ-7: PageRank Graph Algorithm

**User Story**: As a developer, I want to implement PageRank using SpMV to demonstrate practical application of sparse matrix operations on graph data.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 7.1 | Given adjacency matrix and damping factor, should compute PageRank scores using iterative SpMV |
| 7.2 | Should converge when L2 norm of ranking differences between iterations falls below 1e-6 |
| 7.3 | Should handle graphs with dangling nodes (no outgoing edges) correctly |
| 7.4 | Should support graphs with up to 1 million nodes |
| 7.5 | Should output Top-K nodes sorted by ranking score |

**Implementation Status**: ✅ Complete

---

### REQ-8: Error Handling & Resource Management

**User Story**: As a developer, I want robust error handling and resource management so that I can safely use the SpMV library in production code.

#### Acceptance Criteria

| ID | Acceptance Criteria |
|----|---------------------|
| 8.1 | When CUDA memory allocation fails, should return descriptive error and release allocated resources |
| 8.2 | When kernel launch fails, should capture CUDA error and propagate to caller |
| 8.3 | When SpMV operation completes, should synchronize properly and check for asynchronous errors |
| 8.4 | Should provide RAII-style resource management for GPU memory allocation |
| 8.5 | Should validate inputs before GPU operations and return appropriate error codes when given invalid matrix dimensions |

**Implementation Status**: ✅ Complete

---

## Non-Functional Requirements

### Performance Requirements

| Metric | Target | Description |
|--------|--------|-------------|
| Bandwidth Utilization | > 60% | Relative to GPU theoretical peak bandwidth |
| GFLOPS | Proportional to bandwidth | GFLOPS = 2 × nnz / (time × 10⁹) |
| Scalability | Linear growth | Performance grows approximately linearly with nnz |

### Quality Requirements

| Metric | Requirement |
|--------|-------------|
| Code Coverage | Core functionality > 80% |
| Documentation Completeness | All public APIs documented |
| Test Reliability | Property tests pass 100 iterations |

---

## Requirements Traceability Matrix

| Requirement | Design Document Section | Test Properties |
|-------------|------------------------|-----------------|
| REQ-1 | CSR Format | Property 1, 2, 3 |
| REQ-2 | ELL Format | Property 4, 5, 6, 7 |
| REQ-3 | SpMV Kernels | Property 8, 9, 10 |
| REQ-4 | Load Balancing | Property 11 |
| REQ-5 | Bandwidth Optimization | Property 12 |
| REQ-6 | Benchmark Suite | Property 13, 14 |
| REQ-7 | PageRank | Property 15, 16 |
| REQ-8 | Error Handling | All Properties |
