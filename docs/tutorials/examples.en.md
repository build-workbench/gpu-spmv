---
layout: default
title: Examples
nav_order: 5
has_children: false
lang: en
---

<p align="right">
  <a href="examples">🇨🇳 简体中文</a>
</p>

# 📝 Examples
{: .no_toc }

Complete examples from basic usage to advanced applications.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic Examples

### 1. Simple SpMV Computation

```cpp
#include <spmv/spmv.h>
#include <iostream>

int main() {
    // Create 3x3 dense matrix
    float data[] = {
        1, 0, 2,
        0, 3, 4,
        0, 0, 5
    };
    
    // Convert to CSR format
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, data, 3, 3);
    
    // Transfer to GPU
    csr_to_gpu(csr);
    
    // Create input vector
    float h_x[] = {1, 2, 3};
    CudaBuffer<float> d_x(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create output vector
    CudaBuffer<float> d_y(3);
    
    // Execute SpMV
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    
    if (result.error == SpMVError::SUCCESS) {
        std::cout << "SpMV completed in " << result.time_ms << " ms" << std::endl;
    }
    
    // Cleanup
    csr_destroy(csr);
    return 0;
}
```

### 2. Using RAII for Resource Management

```cpp
#include <spmv/spmv.h>

void process_sparse_matrix() {
    // Automatically manage GPU memory
    CudaBuffer<float> d_x(1000);
    CudaBuffer<float> d_y(1000);
    
    // Create CSR matrix
    CSRMatrix* csr = csr_create(100, 100, 500);
    
    // ... populate data ...
    
    csr_to_gpu(csr);
    
    // Execute computation
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 100);
    
    // Automatic cleanup when leaving scope
    csr_destroy(csr);
}
// d_x, d_y automatically released
```

---

## Intermediate Examples

### 3. Multi-Format Support

```cpp
#include <spmv/spmv.h>

void compare_formats() {
    // Create CSR matrix
    CSRMatrix* csr = csr_create(1000, 1000, 5000);
    // ... populate data ...
    csr_to_gpu(csr);
    
    // Convert to ELL format
    ELLMatrix* ell = ell_create(1000, 1000, 
                                csr->row_ptrs[1] - csr->row_ptrs[0]);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
    
    // Test performance of both formats
    CudaBuffer<float> d_x(1000), d_y(1000);
    
    SpMVConfig csr_config = spmv_auto_config(csr);
    SpMVResult csr_result = spmv_csr(csr, d_x.data(), d_y.data(), 
                                     &csr_config, 1000);
    
    SpMVResult ell_result = spmv_ell(ell, d_x.data(), d_y.data(), 1000);
    
    printf("CSR: %.2f ms, ELL: %.2f ms\n", 
           csr_result.time_ms, ell_result.time_ms);
    
    csr_destroy(csr);
    ell_destroy(ell);
}
```

### 4. Error Handling Best Practices

```cpp
#include <spmv/spmv.h>

SpMVError safe_spmv(const CSRMatrix* csr, 
                   const float* d_x, 
                   float* d_y, 
                   int n) {
    if (!csr || !d_x || !d_y) {
        return SpMVError::INVALID_ARGUMENT;
    }
    
    if (csr->num_rows != n || csr->num_cols != n) {
        return SpMVError::INVALID_DIMENSION;
    }
    
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
    
    if (result.error != SpMVError::SUCCESS) {
        fprintf(stderr, "SpMV failed: %s (code: %d)\n", 
                spmv_error_string(result.error), 
                static_cast<int>(result.error));
        return result.error;
    }
    
    printf("SpMV completed in %.3f ms\n", result.time_ms);
    return SpMVError::SUCCESS;
}
```

---

## Advanced Examples

### 5. PageRank Implementation

```cpp
#include <spmv/pagerank.h>
#include <spmv/spmv.h>

void compute_pagerank(const CSRMatrix* adjacency) {
    const int n = adjacency->num_rows;
    
    // Initialize rank vector
    CudaBuffer<float> d_rank(n);
    cudaMemcpy(d_rank.data(), 
               std::vector<float>(n, 1.0f / n).data(), 
               n * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Configure PageRank
    PageRankConfig config;
    config.damping = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;
    
    // Execute PageRank
    SpMVResult result = spmv_pagerank(adjacency, d_rank.data(), &config);
    
    if (result.error == SpMVError::SUCCESS) {
        printf("PageRank converged in %.2f ms\n", result.time_ms);
        
        // Download results
        std::vector<float> h_rank(n);
        cudaMemcpy(h_rank.data(), d_rank.data(), 
                   n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Print top 10 ranks
        for (int i = 0; i < 10; i++) {
            printf("Node %d: %.6f\n", i, h_rank[i]);
        }
    }
}
```

### 6. Performance Benchmarking

```cpp
#include <spmv/benchmark.h>
#include <spmv/spmv.h>

void run_benchmark() {
    // Create test matrix
    CSRMatrix* csr = csr_create(10000, 10000, 500000);
    // ... populate data ...
    csr_to_gpu(csr);
    
    // Configure benchmark
    BenchmarkConfig config;
    config.iterations = 100;
    config.warmup = true;
    config.print_details = true;
    
    // Run benchmark
    spmv_benchmark(csr, &config);
    
    csr_destroy(csr);
}
```

---

## Complete Application Example

### 7. Graph Analysis Application

```cpp
#include <spmv/spmv.h>
#include <spmv/pagerank.h>
#include <iostream>
#include <vector>

class GraphAnalyzer {
public:
    GraphAnalyzer(int num_nodes) : n_(num_nodes) {
        csr_ = csr_create(n_, n_, 0);
    }
    
    ~GraphAnalyzer() {
        if (csr_) csr_destroy(csr_);
    }
    
    void add_edge(int from, int to) {
        // ... add edge to CSR matrix ...
    }
    
    void compute_pagerank() {
        csr_to_gpu(csr_);
        
        CudaBuffer<float> d_rank(n_);
        PageRankConfig config;
        config.damping = 0.85f;
        config.tolerance = 1e-6f;
        config.max_iterations = 100;
        
        SpMVResult result = spmv_pagerank(csr_, d_rank.data(), &config);
        
        if (result.error == SpMVError::SUCCESS) {
            std::vector<float> h_rank(n_);
            cudaMemcpy(h_rank.data(), d_rank.data(), 
                      n_ * sizeof(float), cudaMemcpyDeviceToHost);
            
            print_top_nodes(h_rank, 10);
        }
    }
    
private:
    void print_top_nodes(const std::vector<float>& rank, int top_k) {
        std::vector<std::pair<float, int>> ranked(n_);
        for (int i = 0; i < n_; i++) {
            ranked[i] = {rank[i], i};
        }
        
        std::sort(ranked.begin(), ranked.end(), std::greater<>());
        
        std::cout << "Top " << top_k << " nodes:" << std::endl;
        for (int i = 0; i < top_k && i < n_; i++) {
            std::cout << "  Node " << ranked[i].second 
                     << ": " << ranked[i].first << std::endl;
        }
    }
    
    int n_;
    CSRMatrix* csr_;
};
```

---

## Building and Running

### Compile Examples

```bash
# Build with CMake
cmake --preset release
cmake --build --preset release

# Run tests
./build-release/spmv_tests
```

### Run Benchmarks

```bash
# Run full benchmark
./build-release/spmv_benchmark

# Or use ctest
ctest --preset default
```

---

## Frequently Asked Questions

### Q: How do I choose between CSR and ELL formats?

**A**: 
- Use **CSR**: General purpose, especially when non-zero elements per row vary significantly
- Use **ELL**: When matrix row lengths are uniform and you need extreme performance

### Q: Why is my SpMV performance poor?

**A**: Check the following:
1. Are you using `spmv_auto_config()` to automatically select kernel
2. Has the matrix been transferred to GPU (`csr_to_gpu()`)
3. Are input/output vectors on GPU
4. Is the matrix large enough (small matrices have high overhead ratio)

### Q: How do I handle large-scale matrices?

**A**: 
- Ensure GPU has sufficient memory
- Use `CudaBuffer` for memory management
- Consider chunking for超大 matrices
- Use Merge Path kernel for highly skewed matrices

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>More examples see <a href="https://github.com/LessUp/gpu-spmv/tree/main/benchmarks">benchmarks/</a> directory</p>
</div>
