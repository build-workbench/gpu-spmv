-- -layout : default title : Examples nav_order : 4 permalink : / examples.en lang : en-- -

    <p align = "right"><a href = "examples">🇨🇳 简体中文</ a></ p>
#Examples
    { :.no_toc}

    Complete code examples from basic to advanced.{: .fs-6 .fw-300
}

##Table of Contents {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

   ##Basic Examples

   ## #1. Basic SpMV

```cpp
#include <cstdio>
#include <spmv/spmv.h>

   int
   main() {
    // Create 3x3 sparse matrix [1 0 2; 0 3 4; 0 0 5]
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};

    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, dense, 3, 3);
    csr_to_gpu(csr);

    // Input vector [1, 1, 1]
    float h_x[] = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Execute y = A * x
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);

    if (result.error == SpMVError::SUCCESS) {
        printf("Time: %.3f ms\n", result.time_ms);
        // Result: [3, 7, 5]
    }

    csr_destroy(csr);
    return 0;
}
```

    -- -

    ## #2. Format Conversion(CSR → ELL)

```cpp
    // Convert CSR to ELL for better performance
    CSRMatrix* csr = csr_create(1000, 1000, 5000);
// ... fill CSR data ...

ELLMatrix* ell = ell_create(1000, 1000, 50);
ell_from_csr(ell, csr);
ell_to_gpu(ell);

// ELL is often faster
SpMVResult result = spmv_ell(ell, d_x.data(), d_y.data(), 1000);

ell_destroy(ell);
csr_destroy(csr);
```

    -- -

    ##Intermediate Examples

    ## #3. Error Handling

```cpp SpMVError
    safe_spmv(const CSRMatrix* csr, const float* d_x, float* d_y, int n) {
    if (!csr || !d_x || !d_y) {
        return SpMVError::INVALID_ARGUMENT;
    }

    if (csr->num_rows != n || csr->num_cols != n) {
        return SpMVError::INVALID_DIMENSION;
    }

    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

    if (result.error != SpMVError::SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
    }

    return result.error;
}
```

    -- -

    ## #4. Performance Benchmarking

```cpp
#include <spmv/benchmark.h>

    void
    benchmark_matrix(CSRMatrix* csr) {
    BenchmarkConfig config;
    config.iterations = 100;
    config.warmup = true;
    config.print_details = true;

    spmv_benchmark(csr, &config);
    // Output: Bandwidth, GFLOPS, timing stats
}
```

    -- -

    ##Advanced Examples

    ## #5. PageRank

```cpp
#include <spmv/pagerank.h>

    void
    compute_pagerank(const CSRMatrix* adjacency, int n) {
    // Initialize rank vector
    CudaBuffer<float> d_rank(n);
    std::vector<float> initial(n, 1.0f / n);
    cudaMemcpy(d_rank.data(), initial.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and run
    PageRankConfig config;
    config.damping = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;

    SpMVResult result = pagerank(adjacency, d_rank.data(), &config);

    if (result.error == SpMVError::SUCCESS) {
        printf("PageRank converged in %.2f ms\n", result.time_ms);
    }
}
```

    -- -

    ## #6. Context Reuse

```cpp
    // Reuse execution context for better performance
    void
    batch_spmv(CSRMatrix* csr, const float* d_x, float* d_y, int n, int iterations) {
    SpMVConfig config = spmv_auto_config(csr);

    // Context created outside loop
    for (int i = 0; i < iterations; i++) {
        SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
        // Internal resources are reused
    }
}
```

    -- -

    <div class = "text-center text-small" style = "margin-top: 3rem;">
    <p> More examples in<a href = "https://github.com/LessUp/gpu-spmv/tree/master/benchmarks">
        benchmarks / </ a> directory</ p></ div>
