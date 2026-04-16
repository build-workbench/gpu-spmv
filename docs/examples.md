---
layout: default
title: 示例代码
---

# 示例代码

本页面提供 GPU SpMV 库的完整示例代码，涵盖基本用法、高级功能和实际应用场景。

---

## 目录

- [基本示例](#基本示例)
- [格式转换](#格式转换)
- [执行上下文复用](#执行上下文复用)
- [性能基准测试](#性能基准测试)
- [PageRank 应用](#pagerank-应用)
- [完整应用程序](#完整应用程序)

---

## 基本示例

### 最简 SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"
#include <iostream>
#include <vector>

using namespace spmv;

int main() {
    // 1. 定义稠密矩阵并转换为 CSR 格式
    std::vector<float> dense = {
        1, 0, 2,
        0, 3, 4,
        0, 0, 5
    };
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    
    // 2. 上传到 GPU
    csr_to_gpu(csr);
    
    // 3. 准备输入向量
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);
    
    // 4. 执行 SpMV（自动选择最优 Kernel）
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
    
    // 5. 检查结果
    if (result.error_code != 0) {
        std::cerr << "SpMV failed with error: " << result.error_code << std::endl;
        csr_destroy(csr);
        return 1;
    }
    
    // 6. 获取结果
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    
    std::cout << "Result: ";
    for (float val : y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    // 输出: Result: 3 7 5
    
    std::cout << "Execution time: " << result.elapsed_ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s" << std::endl;
    
    csr_destroy(csr);
    return 0;
}
```

### ELL 格式示例

```cpp
#include "spmv/ell_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

void ell_example() {
    // ELL 格式适合行长度均匀的矩阵
    std::vector<float> dense = {
        1, 0, 2, 0,
        0, 3, 0, 4,
        5, 6, 0, 0,
        0, 0, 7, 8
    };
    
    ELLMatrix* ell = ell_create(0, 0, 0);
    ell_from_dense(ell, dense.data(), 4, 4);
    ell_to_gpu(ell);
    
    std::vector<float> x = {1, 1, 1, 1};
    CudaBuffer<float> d_x(4), d_y(4);
    d_x.copyFromHost(x.data(), 4);
    
    SpMVConfig config;
    config.kernel_type = SpMVConfig::ELL_KERNEL;
    
    SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), &config, 4);
    
    std::vector<float> y(4);
    d_y.copyToHost(y.data(), 4);
    
    ell_destroy(ell);
}
```

---

## 格式转换

### CSR 与 ELL 互转

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/ell_matrix.h"

using namespace spmv;

void format_conversion() {
    // 原始 CSR 矩阵
    CSRMatrix* csr = csr_create(0, 0, 0);
    // ... 填充 CSR 数据
    
    // 计算 ELL 存储效率
    CSRStats stats = csr_compute_stats(csr);
    float fill_ratio = (float)csr->nnz / (csr->num_rows * stats.max_nnz_per_row);
    
    if (fill_ratio > 0.7) {
        // 填充率高，使用 ELL 格式
        ELLMatrix* ell = ell_create(0, 0, 0);
        ell_from_csr(ell, csr);
        
        std::cout << "Converted to ELL format" << std::endl;
        std::cout << "Storage: " << csr->num_rows * stats.max_nnz_per_row << " elements" << std::endl;
        std::cout << "Fill ratio: " << fill_ratio * 100 << "%" << std::endl;
        
        ell_destroy(ell);
    } else {
        std::cout << "CSR format more efficient (fill ratio: " << fill_ratio * 100 << "%)" << std::endl;
    }
    
    csr_destroy(csr);
}
```

### 稀疏矩阵序列化

```cpp
#include "spmv/csr_matrix.h"

using namespace spmv;

void serialization_example() {
    // 创建矩阵
    CSRMatrix* csr = csr_create(1000, 1000, 5000);
    // ... 填充数据
    
    // 序列化到文件
    int ret = csr_serialize(csr, "matrix.bin");
    if (ret != 0) {
        std::cerr << "Serialization failed" << std::endl;
        return;
    }
    std::cout << "Matrix saved to matrix.bin" << std::endl;
    
    csr_destroy(csr);
    
    // 从文件反序列化
    CSRMatrix* loaded = csr_create(0, 0, 0);
    ret = csr_deserialize(loaded, "matrix.bin");
    if (ret != 0) {
        std::cerr << "Deserialization failed" << std::endl;
        return;
    }
    
    std::cout << "Loaded matrix: " << loaded->num_rows << "x" << loaded->num_cols 
              << ", nnz=" << loaded->nnz << std::endl;
    
    csr_destroy(loaded);
}
```

---

## 执行上下文复用

### 纹理缓存复用

```cpp
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

void context_reuse() {
    // 初始化
    CSRMatrix* csr = /* ... */;
    csr_to_gpu(csr);
    
    CudaBuffer<float> d_x(csr->num_cols);
    CudaBuffer<float> d_y(csr->num_rows);
    
    // 配置使用纹理缓存
    SpMVConfig config;
    config.use_texture = true;
    
    // 创建可复用的执行上下文
    SpMVExecutionContext context;
    
    // 多次执行 SpMV，纹理对象只创建一次
    for (int iter = 0; iter < 100; iter++) {
        // 更新输入向量
        std::vector<float> x = /* 生成新输入 */;
        d_x.copyFromHost(x.data(), csr->num_cols);
        
        // 执行 SpMV，context 复用纹理对象
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), 
                                     &config, csr->num_cols, &context);
        
        if (result.error_code != 0) break;
        
        // 使用结果...
        std::cout << "Iteration " << iter << ": " << result.elapsed_ms << " ms" << std::endl;
    }
    
    // context 析构时自动销毁纹理对象
    csr_destroy(csr);
}
```

---

## 性能基准测试

### 完整基准测试示例

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/benchmark.h"
#include "spmv/bandwidth.h"
#include <iostream>
#include <random>

using namespace spmv;

CSRMatrix* generate_random_matrix(int rows, int cols, float density) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> dense(rows * cols, 0.0f);
    for (int i = 0; i < rows * cols; i++) {
        if (dist(rng) < density) {
            dense[i] = dist(rng) * 10.0f;
        }
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), rows, cols);
    return csr;
}

int main() {
    // 生成测试矩阵
    CSRMatrix* csr = generate_random_matrix(10000, 10000, 0.01f);
    csr_to_gpu(csr);
    
    std::cout << "Matrix: " << csr->num_rows << "x" << csr->num_cols 
              << ", nnz=" << csr->nnz << std::endl;
    
    // 生成输入向量
    std::vector<float> x(csr->num_cols, 1.0f);
    
    // 配置基准测试
    BenchmarkConfig bench_config;
    bench_config.num_warmup_runs = 10;
    bench_config.num_runs = 50;
    
    // 测试不同 Kernel
    std::vector<SpMVConfig> configs = {
        {SpMVConfig::SCALAR_CSR, 256, false},
        {SpMVConfig::VECTOR_CSR, 256, false},
        {SpMVConfig::MERGE_PATH, 256, false}
    };
    
    float peak_bandwidth = get_gpu_peak_bandwidth();
    std::cout << "\nGPU Peak Bandwidth: " << peak_bandwidth << " GB/s\n\n";
    
    for (const auto& config : configs) {
        BenchmarkResult result = benchmark_csr(csr, x.data(), &config, &bench_config);
        
        if (result.error_code == 0) {
            std::cout << result.name << ":\n";
            std::cout << "  Avg time: " << result.avg_time_ms << " ms\n";
            std::cout << "  GFLOPS:   " << result.gflops << "\n";
            std::cout << "  Bandwidth: " << result.bandwidth_gb_s << " GB/s ("
                      << (result.bandwidth_gb_s / peak_bandwidth * 100) << "%)\n\n";
        }
    }
    
    // GPU vs CPU 对比
    ComparisonResult comp = compare_gpu_cpu_csr(csr, x.data(), nullptr, &bench_config);
    if (comp.error_code == 0) {
        std::cout << "GPU vs CPU:\n";
        std::cout << "  GPU: " << comp.gpu_result.avg_time_ms << " ms\n";
        std::cout << "  CPU: " << comp.cpu_result.avg_time_ms << " ms\n";
        std::cout << "  Speedup: " << comp.speedup << "x\n";
    }
    
    csr_destroy(csr);
    return 0;
}
```

---

## PageRank 应用

### 基本用法

```cpp
#include "spmv/pagerank.h"
#include "spmv/csr_matrix.h"
#include <iostream>
#include <vector>

using namespace spmv;

int main() {
    // 创建图 (简单的 5 节点图)
    // 0 → 1, 0 → 2
    // 1 → 2, 1 → 3
    // 2 → 0
    // 3 → 4
    // 4 → 2
    
    std::vector<float> adj = {
        0, 1, 1, 0, 0,  // 节点 0 的出边
        0, 0, 1, 1, 0,  // 节点 1 的出边
        1, 0, 0, 0, 0,  // 节点 2 的出边
        0, 0, 0, 0, 1,  // 节点 3 的出边
        0, 0, 1, 0, 0   // 节点 4 的出边
    };
    
    // 列归一化 (每列出边概率)
    int n = 5;
    for (int j = 0; j < n; j++) {
        float col_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            col_sum += adj[i * n + j];
        }
        if (col_sum > 0) {
            for (int i = 0; i < n; i++) {
                adj[i * n + j] /= col_sum;
            }
        }
    }
    
    // 转换为 CSR 并上传 GPU
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, adj.data(), n, n);
    csr_to_gpu(csr);
    
    // 配置 PageRank
    PageRankConfig config;
    config.damping_factor = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;
    
    // 运行 PageRank
    PageRankResult result = pagerank(csr, &config);
    
    if (result.error_code == 0) {
        std::cout << "PageRank converged: " << (result.converged ? "Yes" : "No") << "\n";
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Final residual: " << result.final_residual << "\n\n";
        
        // 获取 Top-K 节点
        int k = 3;
        std::vector<TopKNode> top_k(k);
        pagerank_top_k(&result, n, k, top_k.data());
        
        std::cout << "Top-" << k << " nodes:\n";
        for (int i = 0; i < k; i++) {
            std::cout << "  " << (i + 1) << ". Node " << top_k[i].node_id 
                      << ": " << top_k[i].rank << "\n";
        }
    } else {
        std::cerr << "PageRank failed with error: " << result.error_code << std::endl;
    }
    
    pagerank_free(&result);
    csr_destroy(csr);
    return 0;
}
```

---

## 完整应用程序

### 稀疏矩阵求解器

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace spmv;

// 共轭梯度法求解 Ax = b
class ConjugateGradient {
public:
    ConjugateGradient(int max_iter = 1000, float tol = 1e-6f)
        : max_iter_(max_iter), tol_(tol) {}
    
    std::vector<float> solve(CSRMatrix* A, const std::vector<float>& b) {
        int n = A->num_rows;
        
        // 上传矩阵到 GPU
        csr_to_gpu(A);
        
        // GPU 缓冲区
        CudaBuffer<float> d_x(n), d_r(n), d_p(n), d_Ap(n), d_b(n);
        d_b.copyFromHost(b.data(), n);
        
        // 初始化 x = 0, r = b, p = r
        d_x.memset(0);
        d_r.copyFromHost(b.data(), n);
        d_p.copyFromHost(b.data(), n);
        
        // 执行上下文
        SpMVConfig config = spmv_auto_config(A);
        SpMVExecutionContext context;
        
        // 计算 r·r
        float rsold = dot_product(d_r.get(), d_r.get(), n);
        
        for (int iter = 0; iter < max_iter_; iter++) {
            // Ap = A * p
            spmv_csr(A, d_p.get(), d_Ap.get(), &config, n, &context);
            
            // alpha = rsold / (p · Ap)
            float pAp = dot_product(d_p.get(), d_Ap.get(), n);
            float alpha = rsold / pAp;
            
            // x = x + alpha * p
            axpy(d_x.get(), d_p.get(), alpha, n);
            
            // r = r - alpha * Ap
            axpy(d_r.get(), d_Ap.get(), -alpha, n);
            
            // 检查收敛
            float rsnew = dot_product(d_r.get(), d_r.get(), n);
            if (std::sqrt(rsnew) < tol_) {
                std::cout << "Converged at iteration " << iter << std::endl;
                break;
            }
            
            // p = r + (rsnew/rsold) * p
            float beta = rsnew / rsold;
            xpay(d_p.get(), d_r.get(), beta, n);
            
            rsold = rsnew;
        }
        
        // 下载结果
        std::vector<float> x(n);
        d_x.copyToHost(x.data(), n);
        return x;
    }
    
private:
    int max_iter_;
    float tol_;
    
    // 辅助函数 (简化版，实际应使用 cuBLAS)
    float dot_product(const float* a, const float* b, int n);
    void axpy(float* y, const float* x, float alpha, int n);
    void xpay(float* y, const float* x, float beta, int n);
};

int main() {
    // 创建测试系统
    CSRMatrix* A = /* 创建对称正定矩阵 */;
    std::vector<float> b = /* 创建右端项 */;
    
    ConjugateGradient solver(1000, 1e-6f);
    std::vector<float> x = solver.solve(A, b);
    
    // 输出结果
    std::cout << "Solution: ";
    for (float val : x) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    csr_destroy(A);
    return 0;
}
```
