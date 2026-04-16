---
layout: default
title: Examples
lang: en
---

<p align="right">
  <a href="examples.html">🇨🇳 简体中文</a>
</p>

# 📝 Code Examples

This page provides complete code examples for GPU SpMV library, covering basic usage, advanced features, and real-world applications.

---

## Table of Contents

- [Basic Examples](#basic-examples)
- [Format Conversion](#format-conversion)
- [Context Reuse](#context-reuse)
- [Benchmarking](#benchmarking)
- [PageRank Application](#pagerank-application)

---

## Basic Examples

### Minimal SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"
#include <iostream>
#include <vector>

using namespace spmv;

int main() {
    // 1. Define dense matrix and convert to CSR
    std::vector<float> dense = {
        1, 0, 2,
        0, 3, 4,
        0, 0, 5
    };
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);
    
    // 2. Prepare vectors
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);
    
    // 3. Execute SpMV (auto-select kernel)
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
    
    // 4. Retrieve results
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    
    std::cout << "Result: " << y[0] << " " << y[1] << " " << y[2] << std::endl;
    std::cout << "Time: " << result.elapsed_ms << " ms" << std::endl;
    
    csr_destroy(csr);
    return 0;
}
```

---

## Format Conversion

### CSR to ELL Conversion

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/ell_matrix.h"

void format_conversion() {
    CSRMatrix* csr = /* ... */;
    
    // Calculate ELL storage efficiency
    CSRStats stats = csr_compute_stats(csr);
    float fill_ratio = (float)csr->nnz / (csr->num_rows * stats.max_nnz_per_row);
    
    if (fill_ratio > 0.7) {
        ELLMatrix* ell = ell_create(0, 0, 0);
        ell_from_csr(ell, csr);
        // Use ELL for better GPU performance
        ell_destroy(ell);
    }
    
    csr_destroy(csr);
}
```

---

## Context Reuse

### Texture Cache Reuse

```cpp
#include "spmv/spmv.h"

void context_reuse() {
    CSRMatrix* csr = /* ... */;
    csr_to_gpu(csr);
    
    CudaBuffer<float> d_x(csr->num_cols);
    CudaBuffer<float> d_y(csr->num_rows);
    
    SpMVConfig config;
    config.use_texture = true;
    
    // Create reusable context
    SpMVExecutionContext context;
    
    // Multiple executions with texture reuse
    for (int iter = 0; iter < 100; iter++) {
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), 
                                     &config, csr->num_cols, &context);
        // Texture object created only once
    }
    
    csr_destroy(csr);
}
```

---

## Benchmarking

### Full Benchmark Example

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/benchmark.h"
#include <iostream>

using namespace spmv;

int main() {
    // Generate or load matrix
    CSRMatrix* csr = /* ... */;
    csr_to_gpu(csr);
    
    std::vector<float> x(csr->num_cols, 1.0f);
    
    // Benchmark configuration
    BenchmarkConfig bench_config;
    bench_config.num_warmup_runs = 10;
    bench_config.num_runs = 50;
    
    // Run benchmark
    BenchmarkResult result = benchmark_csr(csr, x.data(), nullptr, &bench_config);
    
    if (result.error_code == 0) {
        std::cout << "Average time: " << result.avg_time_ms << " ms\n";
        std::cout << "GFLOPS: " << result.gflops << "\n";
        std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";
    }
    
    csr_destroy(csr);
    return 0;
}
```

---

## PageRank Application

### Basic PageRank

```cpp
#include "spmv/pagerank.h"
#include "spmv/csr_matrix.h"
#include <iostream>
#include <vector>

using namespace spmv;

int main() {
    // Create and normalize adjacency matrix
    CSRMatrix* adj = /* ... */;
    csr_to_gpu(adj);
    
    // Configure PageRank
    PageRankConfig config;
    config.damping_factor = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;
    
    // Run PageRank
    PageRankResult result = pagerank(adj, &config);
    
    if (result.error_code == 0) {
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        
        // Get Top-K nodes
        int k = 10;
        std::vector<TopKNode> top_k(k);
        pagerank_top_k(&result, adj->num_rows, k, top_k.data());
        
        std::cout << "Top-" << k << " nodes:\n";
        for (int i = 0; i < k; i++) {
            std::cout << "  " << (i+1) << ". Node " << top_k[i].node_id 
                      << ": " << top_k[i].rank << "\n";
        }
    }
    
    pagerank_free(&result);
    csr_destroy(adj);
    return 0;
}
```

---

<div align="center">

**[← API Reference](api.en)** · **[ Performance →](performance.en)**

</div>
