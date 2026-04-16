---
layout: default
title: Performance Optimization
lang: en
---

<p align="right">
  <a href="performance.html">🇨🇳 简体中文</a>
</p>

# 🚀 Performance Optimization Guide

This guide covers performance optimization strategies, kernel selection mechanisms, and benchmarking best practices for GPU SpMV.

---

## Table of Contents

- [Performance Overview](#performance-overview)
- [Kernel Selection](#kernel-selection)
- [Memory Bandwidth](#memory-bandwidth)
- [Benchmarking](#benchmarking)
- [Optimization Tips](#optimization-tips)

---

## Performance Overview

### Memory-Bound Nature

SpMV is fundamentally **memory-bandwidth-bound**:

```
Computation: 2 × nnz FLOPs (multiply-add per non-zero)
Data Movement: ~20 × nnz bytes (values + indices + vectors)

Arithmetic Intensity: ~0.1 FLOP/Byte (far below GPU capability)
```

Therefore, SpMV optimization focuses on **maximizing memory bandwidth utilization**.

### Performance Targets

| Metric | Target | Notes |
|:-------|:-------|:------|
| Bandwidth Utilization | > 60% | Relative to GPU theoretical peak |
| GFLOPS | Bandwidth-proportional | GFLOPS = 2 × nnz / (time × 10^9) |
| Scalability | Linear | Performance grows linearly with nnz |

---

## Kernel Selection

SpMV performance heavily depends on non-zero distribution. This library provides intelligent kernel selection.

### Decision Tree

```
                 Matrix Analysis
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────┴────┐              ┌─────┴──────┐
    │ FORMAT  │              │ Statistics │
    └────┬────┘              └─────┬──────┘
         │                         │
    ELL ─┼─→ ELL Kernel    ┌──────┴──────┐
         │                 │             │
        CSR          avg_nnz < 4?   skewness < 10?
                            │             │
                          Yes            No
                            │             │
                       ┌────┴────┐   ┌────┴────┐
                       │ Scalar  │   │ Vector  │
                       │   CSR   │   │   CSR   │
                       └─────────┘   └────┬────┘
                                          │
                                         No
                                          │
                                     ┌────┴────┐
                                     │  Merge  │
                                     │  Path   │
                                     └─────────┘
```

### Kernel Comparison

| Kernel | Strategy | Sync Overhead | Load Balance | Best For |
|:-------|:---------|:-------------:|:------------:|:---------|
| **Scalar CSR** | 1 thread/row | None | Poor | avg_nnz < 4 |
| **Vector CSR** | 1 warp/row | Warp shuffle | Medium | Uniform distribution |
| **Merge Path** | Equal partitioning | Atomic ops | Perfect | Highly irregular |
| **ELL** | Column-major | None | - | ELL format matrices |

---

## Memory Bandwidth

### Coalesced Access

ELL Column-major storage enables fully coalesced access:

```
Row-major (poor):
Thread:  T0      T1      T2
         ↓       ↓       ↓
Addr: [r0,k0][r1,k0][r2,k0]  ← Discontinuous

Column-major (good):
Thread:  T0      T1      T2
         ↓       ↓       ↓
Addr: [r0,k0][r1,k0][r2,k0]  ← Continuous!
       [base+0] [base+1] [base+2]
```

### Texture Cache

Random access to input vector `x` (determined by column indices) benefits from texture cache:

```cpp
SpMVConfig config = spmv_auto_config(csr);
config.use_texture = true;  // Enable texture cache

// Or auto-enabled when num_cols > 10000
```

**When to Use:**
- `num_cols > 10000`: Large vectors
- Obvious random access patterns

**Best Practice:**
- Use `SpMVExecutionContext` to reuse texture objects
- Avoid frequent creation/destruction

### Warp Shuffle Reduction

Vector CSR uses shuffle instructions for reduction, avoiding shared memory bank conflicts:

```cpp
// Traditional shared memory reduction (potential bank conflicts)
__shared__ float sdata[32];
sdata[lane_id] = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    sdata[lane_id] += sdata[lane_id + offset];  // May conflict
}

// Shuffle reduction (no bank conflicts)
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);  // Fully parallel
}
```

---

## Benchmarking

### Running Benchmarks

```bash
# Build and run
cmake --preset release && cmake --build --preset release
./build-release/spmv_benchmark
```

### Sample Output

```
========================================
GPU SpMV Benchmark
========================================
GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Memory: 10240 MB
Memory Bandwidth: 760.3 GB/s

Matrix: 1000x1000, NNZ: 50000, Density: 0.05

Scalar CSR:
  Avg time: 0.042 ms
  Min time: 0.038 ms
  Max time: 0.051 ms
  GFLOPS: 2381.0
  Bandwidth: 125.3 GB/s (16.5%)

Vector CSR:
  Avg time: 0.031 ms
  GFLOPS: 3225.8
  Bandwidth: 169.5 GB/s (22.3%)

Merge Path:
  Avg time: 0.035 ms
  GFLOPS: 2857.1
  Bandwidth: 150.2 GB/s (19.8%)
```

### Metrics Explanation

| Metric | Calculation | Meaning |
|:-------|:------------|:--------|
| `avg_time_ms` | Average execution time | Excluding warmup runs |
| `gflops` | `2 × nnz / (time × 10^9)` | Floating-point operations per second |
| `bandwidth_gb_s` | `bytes / (time × 10^9)` | Actual memory bandwidth |
| `efficiency` | `achieved / theoretical` | Bandwidth utilization ratio |

---

## Optimization Tips

### 1. Choose Appropriate Matrix Format

```cpp
// Calculate ELL storage efficiency
float ell_efficiency = (float)nnz / (num_rows * max_nnz_per_row);

if (ell_efficiency > 0.8) {
    // Use ELL format
    ELLMatrix* ell = ell_create(0, 0, 0);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
} else {
    // Use CSR format
    csr_to_gpu(csr);
}
```

### 2. Enable Texture Cache

```cpp
// Enable texture cache for large vectors
SpMVConfig config = spmv_auto_config(csr);
if (csr->num_cols > 10000) {
    config.use_texture = true;
}
```

### 3. Reuse GPU Memory

```cpp
// Avoid repeated allocations
CudaBuffer<float> d_x(cols);
CudaBuffer<float> d_y(rows);

for (int iter = 0; iter < iterations; iter++) {
    d_x.copyFromHost(new_x.data(), cols);
    spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
    d_y.copyToHost(result.data(), rows);
}
```

### 4. Reuse Execution Context

```cpp
// Reuse texture objects across calls
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < 100; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
// Texture object automatically destroyed with context
```

### 5. Check Bandwidth Utilization

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);

float peak = get_gpu_peak_bandwidth();
float efficiency = result.bandwidth_gb_s / peak;

printf("Bandwidth utilization: %.1f%%\n", efficiency * 100);

if (efficiency < 0.5) {
    printf("Consider:\n");
    printf("  - Using ELL format for uniform row lengths\n");
    printf("  - Enabling texture cache for large vectors\n");
}
```

---

<div align="center">

**[← Examples](examples.en)** · **[ Changelog →](changelog.en)**

</div>
