#include "spmv/csr_matrix.h"
#include "spmv/ell_matrix.h"
#include "spmv/spmv.h"
#include "spmv/benchmark.h"
#include "spmv/pagerank.h"
#include <iostream>
#include <vector>
#include <random>

using namespace spmv;

void print_separator() {
    std::cout << "========================================\n";
}

void benchmark_spmv() {
    print_separator();
    std::cout << "SpMV Benchmark\n";
    print_separator();
    
    // 创建测试矩阵
    int rows = 1000;
    int cols = 1000;
    float density = 0.05f;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> dense(rows * cols, 0.0f);
    for (int i = 0; i < rows * cols; i++) {
        if (dist(rng) < density) {
            dense[i] = dist(rng) * 10.0f;
        }
    }
    
    std::vector<float> x(cols, 1.0f);
    
    // CSR 格式
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), rows, cols);
    csr_to_gpu(csr);
    
    std::cout << "Matrix: " << rows << "x" << cols 
              << ", NNZ: " << csr->nnz 
              << ", Density: " << (float)csr->nnz / (rows * cols) << "\n\n";
    
    // 测试不同 Kernel
    BenchmarkConfig bench_config;
    bench_config.num_warmup_runs = 5;
    bench_config.num_runs = 20;
    
    SpMVConfig configs[] = {
        {SpMVConfig::SCALAR_CSR, 256, false},
        {SpMVConfig::VECTOR_CSR, 256, false},
        {SpMVConfig::MERGE_PATH, 256, false}
    };
    
    const char* names[] = {"Scalar CSR", "Vector CSR", "Merge Path"};
    
    for (int i = 0; i < 3; i++) {
        BenchmarkResult result = benchmark_csr(csr, x.data(), &configs[i], &bench_config);
        
        std::cout << names[i] << ":\n";
        std::cout << "  Avg time: " << result.avg_time_ms << " ms\n";
        std::cout << "  Min time: " << result.min_time_ms << " ms\n";
        std::cout << "  Max time: " << result.max_time_ms << " ms\n";
        std::cout << "  Stddev: " << result.stddev_time_ms << " ms\n";
        std::cout << "  GFLOPS: " << result.gflops << "\n";
        std::cout << "  Bandwidth: " << result.bandwidth_gb_s << " GB/s\n\n";
    }
    
    // GPU vs CPU 对比
    std::cout << "GPU vs CPU Comparison:\n";
    ComparisonResult comp = compare_gpu_cpu_csr(csr, x.data(), nullptr, &bench_config);
    std::cout << "  GPU time: " << comp.gpu_result.avg_time_ms << " ms\n";
    std::cout << "  CPU time: " << comp.cpu_result.avg_time_ms << " ms\n";
    std::cout << "  Speedup: " << comp.speedup << "x\n\n";
    
    csr_destroy(csr);
}

void benchmark_pagerank() {
    print_separator();
    std::cout << "PageRank Benchmark\n";
    print_separator();
    
    // 创建随机图
    int n = 100;
    float density = 0.1f;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> adj(n * n, 0.0f);
    for (int i = 0; i < n * n; i++) {
        if (dist(rng) < density) {
            adj[i] = dist(rng);
        }
    }
    
    // 列归一化
    for (int j = 0; j < n; j++) {
        float col_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            col_sum += adj[i * n + j];
        }
        if (col_sum > 0.0f) {
            for (int i = 0; i < n; i++) {
                adj[i * n + j] /= col_sum;
            }
        }
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, adj.data(), n, n);
    csr_to_gpu(csr);
    
    std::cout << "Graph: " << n << " nodes, " << csr->nnz << " edges\n\n";
    
    PageRankConfig config;
    config.damping_factor = 0.85f;
    config.tolerance = 1e-6f;
    config.max_iterations = 100;
    
    PageRankResult result = pagerank(csr, &config);
    
    std::cout << "PageRank Results:\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Converged: " << (result.converged ? "Yes" : "No") << "\n";
    std::cout << "  Final residual: " << result.final_residual << "\n\n";
    
    // Top-10 节点
    std::vector<TopKNode> top_10(10);
    pagerank_top_k(&result, n, 10, top_10.data());
    
    std::cout << "Top-10 Nodes:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  " << (i + 1) << ". Node " << top_10[i].node_id 
                  << ": " << top_10[i].rank << "\n";
    }
    
    pagerank_free(&result);
    csr_destroy(csr);
}

int main() {
    std::cout << "\nGPU SpMV Benchmark Suite\n";
    print_separator();
    
    // 打印 GPU 信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Memory Bandwidth: " << get_gpu_peak_bandwidth() << " GB/s\n\n";
    
    benchmark_spmv();
    benchmark_pagerank();
    
    print_separator();
    std::cout << "Benchmark Complete!\n";
    print_separator();
    
    return 0;
}
