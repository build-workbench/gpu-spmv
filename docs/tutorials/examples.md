---
layout: default
title: 示例代码
nav_order: 4
has_children: false
lang: zh
---

<p align="right">
  <a href="examples.en">🇺🇸 English</a>
</p>

# 📝 示例代码
{: .no_toc }

从基础用法到高级应用的完整示例。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 基础示例

### 1. 简单的 SpMV 计算

```cpp
#include <spmv/spmv.h>
#include <iostream>

int main() {
    // 创建 3x3 密集矩阵
    float data[] = {
        1, 0, 2,
        0, 3, 4,
        0, 0, 5
    };
    
    // 转换为 CSR 格式
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, data, 3, 3);
    
    // 传输到 GPU
    csr_to_gpu(csr);
    
    // 创建输入向量
    float h_x[] = {1, 2, 3};
    CudaBuffer<float> d_x(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // 创建输出向量
    CudaBuffer<float> d_y(3);
    
    // 执行 SpMV
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    
    if (result.error == SpMVError::SUCCESS) {
        std::cout << "SpMV completed in " << result.time_ms << " ms" << std::endl;
    }
    
    // 清理
    csr_destroy(csr);
    return 0;
}
```

### 2. 使用 RAII 管理资源

```cpp
#include <spmv/spmv.h>

void process_sparse_matrix() {
    // 自动管理 GPU 内存
    CudaBuffer<float> d_x(1000);
    CudaBuffer<float> d_y(1000);
    
    // 创建 CSR 矩阵
    CSRMatrix* csr = csr_create(100, 100, 500);
    
    // ... 填充数据 ...
    
    csr_to_gpu(csr);
    
    // 执行计算
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 100);
    
    // 离开作用域时自动清理
    csr_destroy(csr);
}
// d_x, d_y 自动释放
```

---

## 中级示例

### 3. 多格式支持

```cpp
#include <spmv/spmv.h>

void compare_formats() {
    // 创建 CSR 矩阵
    CSRMatrix* csr = csr_create(1000, 1000, 5000);
    // ... 填充数据 ...
    csr_to_gpu(csr);
    
    // 转换为 ELL 格式
    ELLMatrix* ell = ell_create(1000, 1000, 
                                csr->row_ptrs[1] - csr->row_ptrs[0]);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
    
    // 测试两种格式的性能
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

### 4. 错误处理最佳实践

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

## 高级示例

### 5. PageRank 实现

```cpp
#include <spmv/pagerank.h>
#include <spmv/spmv.h>

void compute_pagerank(const CSRMatrix* adjacency) {
    const int n = adjacency->num_rows;
    
    // 初始化排名向量
    CudaBuffer<float> d_rank(n);
    cudaMemcpy(d_rank.data(), 
               std::vector<float>(n, 1.0f / n).data(), 
               n * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 配置 PageRank
    PageRankConfig config;
    config.damping = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;
    
    // 执行 PageRank
    SpMVResult result = spmv_pagerank(adjacency, d_rank.data(), &config);
    
    if (result.error == SpMVError::SUCCESS) {
        printf("PageRank converged in %.2f ms\n", result.time_ms);
        
        // 下载结果
        std::vector<float> h_rank(n);
        cudaMemcpy(h_rank.data(), d_rank.data(), 
                   n * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 打印前 10 个排名
        for (int i = 0; i < 10; i++) {
            printf("Node %d: %.6f\n", i, h_rank[i]);
        }
    }
}
```

### 6. 性能基准测试

```cpp
#include <spmv/benchmark.h>
#include <spmv/spmv.h>

void run_benchmark() {
    // 创建测试矩阵
    CSRMatrix* csr = csr_create(10000, 10000, 500000);
    // ... 填充数据 ...
    csr_to_gpu(csr);
    
    // 配置基准测试
    BenchmarkConfig config;
    config.iterations = 100;
    config.warmup = true;
    config.print_details = true;
    
    // 运行基准测试
    spmv_benchmark(csr, &config);
    
    csr_destroy(csr);
}
```

---

## 完整应用示例

### 7. 图分析应用

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
        // ... 添加边到 CSR 矩阵 ...
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

## 构建和运行

### 编译示例

```bash
# 使用 CMake 构建
cmake --preset release
cmake --build --preset release

# 运行测试
./build-release/spmv_tests
```

### 运行基准测试

```bash
# 运行完整基准测试
./build-release/spmv_benchmark

# 或使用 ctest
ctest --preset default
```

---

## 常见问题

### Q: 如何选择 CSR 和 ELL 格式？

**A**: 
- 使用 **CSR**: 通用场景，特别是当每行非零元素数差异较大时
- 使用 **ELL**: 当矩阵行长度均匀，且需要极致性能时

### Q: 为什么我的 SpMV 性能不佳？

**A**: 检查以下几点：
1. 是否使用了 `spmv_auto_config()` 自动选择 kernel
2. 矩阵是否已传输到 GPU (`csr_to_gpu()`)
3. 输入输出向量是否在 GPU 上
4. 矩阵规模是否足够大（小规模矩阵 overhead 占比高）

### Q: 如何处理大规模矩阵？

**A**: 
- 确保 GPU 有足够内存
- 使用 `CudaBuffer` 管理内存
- 考虑分块处理超大矩阵
- 使用 Merge Path kernel 处理高度倾斜的矩阵

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>更多示例见 <a href="https://github.com/LessUp/gpu-spmv/tree/main/benchmarks">benchmarks/</a> 目录</p>
</div>
