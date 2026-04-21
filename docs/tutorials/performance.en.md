---
layout: default
title: Performance Optimization
nav_order: 6
has_children: false
lang: en
---

<p align="right">
  <a href="performance">🇨🇳 简体中文</a>
</p>

# 🚀 Performance Optimization
{: .no_toc }

Tuning strategies, benchmark testing, and best practices.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Performance Overview

GPU SpMV library achieves extreme performance through multiple kernel intelligent scheduling.

### Core Performance Metrics

| Metric | Description | Target |
|:-------|:-----------|:-------|
| **Bandwidth Utilization** | Actual memory bandwidth / theoretical peak | > 60% |
| **Compute Density** | FLOPS / byte access | Matrix-dependent |
| **Scalability** | Performance growth with matrix size | Linear scaling |

---

## Kernel Selection Strategy

### 1. Scalar CSR Kernel

**Use Case**: Very sparse matrices (avg_nnz < 4)

```cpp
// Auto-selection
SpMVConfig config = spmv_auto_config(csr);
// config.kernel_type == KernelType::SCALAR_CSR
```

**Performance Characteristics**:
- Each thread processes one row
- Minimizes inter-thread coordination overhead
- Suitable for cases with very few non-zero elements

**Bandwidth Utilization**: ~40-50%

### 2. Vector CSR Kernel

**Use Case**: Moderate sparsity matrices (skewness < 10)

**Performance Characteristics**:
- Each warp collaboratively processes one row
- Coalesced memory access pattern
- Balanced load distribution

**Bandwidth Utilization**: ~65-75%

### 3. Merge Path Kernel

**Use Case**: Highly skewed matrices (skewness ≥ 10)

**Performance Characteristics**:
- Perfect load balancing
- Binary search partition points
- Adaptive to matrix features

**Bandwidth Utilization**: ~70-80%

### 4. ELL Kernel

**Use Case**: ELL format matrices

**Performance Characteristics**:
- Fully coalesced memory access
- Column-major storage
- Highest bandwidth utilization

**Bandwidth Utilization**: ~80-90%

---

## Performance Tuning Guide

### 1. Auto Configuration (Recommended)

```cpp
// Let the library automatically select optimal kernel
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

**Advantages**:
- No manual tuning required
- Intelligent selection based on matrix features
- Suitable for most scenarios

### 2. Manual Kernel Selection

```cpp
// Manually select for specific scenarios
SpMVConfig config;
config.kernel_type = KernelType::MERGE_PATH;
config.auto_select = false;

SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

**Use Cases**:
- Known stable matrix features
- Need extreme performance
- Auto-selection results not ideal

### 3. Format Conversion

```cpp
// CSR -> ELL conversion
ELLMatrix* ell = ell_create(num_rows, num_cols, max_nnz_per_row);
ell_from_csr(ell, csr);
ell_to_gpu(ell);

// ELL format usually performs better
SpMVResult result = spmv_ell(ell, d_x, d_y, n);
```

**When to Convert**:
- Matrix row lengths are uniform
- Non-zero elements per row variation < 20%
- Pursuing extreme performance

---

## Benchmarking

### Running Benchmarks

```cpp
#include <spmv/benchmark.h>

BenchmarkConfig config;
config.iterations = 100;      // 100 iterations
config.warmup = true;         // Warmup
config.print_details = true;  // Detailed information

spmv_benchmark(csr, &config);
```

### Example Output

```
=== GPU SpMV Benchmark ===
Matrix: 10000 x 10000, nnz = 500000
Kernel: Vector CSR
Iterations: 100 (10 warmup)

Results:
  Avg:  2.34 ms
  Min:  2.12 ms
  Max:  2.89 ms
  Std:  0.15 ms

Bandwidth: 68.5 GB/s (70.2% of peak)
GFLOPS: 42.8
```

### Custom Benchmark

```cpp
#include <spmv/spmv.h>
#include <chrono>

void custom_benchmark(const CSRMatrix* csr, 
                     const float* d_x, 
                     float* d_y, 
                     int n,
                     int iterations) {
    // Warmup
    SpMVConfig config = spmv_auto_config(csr);
    for (int i = 0; i < 5; i++) {
        spmv_csr(csr, d_x, d_y, &config, n);
    }
    
    // Official test
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        spmv_csr(csr, d_x, d_y, &config, n);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count();
    
    printf("Avg: %.3f ms\n", duration / 1000.0 / iterations);
}
```

---

## Performance Optimization Best Practices

### 1. Memory Optimization

```cpp
// ✅ Recommended: Use RAII
void process() {
    CudaBuffer<float> d_x(1000000);
    CudaBuffer<float> d_y(1000000);
    // Automatic lifecycle management
}

// ❌ Avoid: Manual management
void process() {
    float *d_x, *d_y;
    cudaMalloc(&d_x, 1000000 * sizeof(float));
    cudaMalloc(&d_y, 1000000 * sizeof(float));
    // Easy to forget cudaFree
    cudaFree(d_x);
    cudaFree(d_y);
}
```

### 2. Execution Context Reuse

```cpp
// ✅ Recommended: Reuse context
SpMVExecutionContext ctx;
for (int i = 0; i < 100; i++) {
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
    // Texture objects and cache configuration are reused
}

// ❌ Avoid: Create each time
for (int i = 0; i < 100; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
    // Repeatedly create texture objects
}
```

### 3. Batch Processing

```cpp
// ✅ Recommended: Batch process multiple matrices
void process_batch(const std::vector<CSRMatrix*>& matrices) {
    for (auto* csr : matrices) {
        csr_to_gpu(csr);
        SpMVConfig config = spmv_auto_config(csr);
        spmv_csr(csr, d_x, d_y, &config, csr->num_rows);
    }
}

// ❌ Avoid: Process one by one
for (int i = 0; i < 100; i++) {
    CSRMatrix* csr = load_matrix(i);
    csr_to_gpu(csr);
    spmv_csr(csr, d_x, d_y, &config, n);
    csr_destroy(csr);
}
```

### 4. Data Transfer Optimization

```cpp
// ✅ Recommended: Async transfer
cudaMemcpyAsync(d_x.data(), h_x.data(), 
                n * sizeof(float), 
                cudaMemcpyHostToDevice, 
                stream);

// ❌ Avoid: Sync transfer blocks
cudaMemcpy(d_x.data(), h_x.data(), 
           n * sizeof(float), 
           cudaMemcpyHostToDevice);
```

---

## Benchmark Data

### NVIDIA RTX 3090 (Ampere) Test Results

| Matrix Size | Non-Zero Elements | Kernel | Time (ms) | Bandwidth (GB/s) | Utilization |
|:-----------:|:-----------------:|:-------|:---------:|:----------------:|:-----------:|
| 10K × 10K | 500K | Vector CSR | 2.34 | 68.5 | 70.2% |
| 50K × 50K | 2.5M | Merge Path | 11.8 | 71.2 | 72.9% |
| 100K × 100K | 5M | Merge Path | 23.5 | 69.8 | 71.5% |
| 500K × 500K | 25M | Merge Path | 118.3 | 70.5 | 72.2% |
| 1M × 1M | 50M | Merge Path | 235.7 | 69.1 | 70.8% |

### Different GPU Architecture Comparison

| GPU Architecture | Representative Model | Theoretical Bandwidth | Actual Utilization |
|:----------------|:--------------------|:---------------------:|:------------------:|
| Volta | V100 | 900 GB/s | ~65% |
| Turing | RTX 2080 | 448 GB/s | ~68% |
| Ampere | RTX 3090 | 936 GB/s | ~70% |
| Ada Lovelace | RTX 4090 | 1008 GB/s | ~72% |

---

## Troubleshooting

### Common Reasons for Poor Performance

1. **Not using auto configuration**
   ```cpp
   // ❌ Wrong
   SpMVConfig config;
   config.kernel_type = KernelType::SCALAR_CSR;  // Manually selected inefficient kernel
   ```

2. **Data not transferred to GPU**
   ```cpp
   // ❌ Wrong
   csr_to_gpu(csr);  // Forgot to call
   spmv_csr(csr, d_x, d_y, &config, n);  // Executing on CPU data
   ```

3. **Matrix size too small**
   ```cpp
   // Warning: 100x100 matrix has high overhead ratio
   CSRMatrix* csr = csr_create(100, 100, 500);
   ```

4. **Frequent memory allocation**
   ```cpp
   // ❌ Wrong: Allocate in loop
   for (int i = 0; i < 100; i++) {
       CudaBuffer<float> buf(1000);  // Allocate each iteration
   }
   ```

### Performance Analysis Tools

```bash
# Use nvprof for analysis
nvprof ./spmv_benchmark

# Use Nsight Systems
nsys profile ./spmv_benchmark

# Use Nsight Compute
ncu --kernel-name spmv ./spmv_benchmark
```

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>Complete performance data see <a href="https://github.com/LessUp/gpu-spmv/tree/main/benchmarks">benchmarks/</a> directory</p>
</div>
