-- -layout : default title : 示例代码 nav_order : 4 permalink : / examples lang : zh-- -

    <p align = "right"><a href = "examples.en">🇺🇸 English</ a></ p>
#示例代码
    { :.no_toc}

    从基础到高级的完整代码示例。 {: .fs-6 .fw-300
}

##目录 {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

   ##基础示例

   ## #1. 基础 SpMV

```cpp
#include <cstdio>
#include <spmv/spmv.h>

   int
   main() {
    // 创建 3x3 稀疏矩阵 [1 0 2; 0 3 4; 0 0 5]
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};

    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, dense, 3, 3);
    csr_to_gpu(csr);

    // 输入向量 [1, 1, 1]
    float h_x[] = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // 执行 y = A * x
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);

    if (result.error == SpMVError::SUCCESS) {
        printf("Time: %.3f ms\n", result.time_ms);
        // 结果: [3, 7, 5]
    }

    csr_destroy(csr);
    return 0;
}
```

    -- -

    ## #2. 格式转换（CSR → ELL）

```cpp
    // 将 CSR 转换为 ELL 以获得更好性能
    CSRMatrix* csr = csr_create(1000, 1000, 5000);
// ... 填充 CSR 数据 ...

ELLMatrix* ell = ell_create(1000, 1000, 50);
ell_from_csr(ell, csr);
ell_to_gpu(ell);

// ELL 通常更快
SpMVResult result = spmv_ell(ell, d_x.data(), d_y.data(), 1000);

ell_destroy(ell);
csr_destroy(csr);
```

    -- -

    ##中级示例

    ## #3. 错误处理

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

    ## #4. 性能测试

```cpp
#include <spmv/benchmark.h>

    void
    benchmark_matrix(CSRMatrix* csr) {
    BenchmarkConfig config;
    config.iterations = 100;
    config.warmup = true;
    config.print_details = true;

    spmv_benchmark(csr, &config);
    // 输出: Bandwidth, GFLOPS, timing stats
}
```

    -- -

    ##高级示例

    ## #5. PageRank

```cpp
#include <spmv/pagerank.h>

    void
    compute_pagerank(const CSRMatrix* adjacency, int n) {
    // 初始化排名向量
    CudaBuffer<float> d_rank(n);
    std::vector<float> initial(n, 1.0f / n);
    cudaMemcpy(d_rank.data(), initial.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 配置并运行
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

    ## #6. 上下文复用

```cpp
    // 复用执行上下文提高性能
    void
    batch_spmv(CSRMatrix* csr, const float* d_x, float* d_y, int n, int iterations) {
    SpMVConfig config = spmv_auto_config(csr);

    // 在循环外创建上下文
    for (int i = 0; i < iterations; i++) {
        SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
        // 内部资源被复用
    }
}
```

    -- -

    <div class = "text-center text-small" style = "margin-top: 3rem;">
    <p> 更多示例见<a href = "https://github.com/LessUp/gpu-spmv/tree/master/benchmarks">
        benchmarks / </ a> 目录</ p></ div>
