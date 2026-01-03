#ifndef SPMV_BENCHMARK_H
#define SPMV_BENCHMARK_H

#include "spmv.h"
#include "csr_matrix.h"
#include "ell_matrix.h"
#include <string>
#include <vector>

namespace spmv {

// 基准测试结果
struct BenchmarkResult {
    std::string name;           // 测试名称
    float execution_time_ms;    // 执行时间 (ms)
    float gflops;               // 计算吞吐量
    float bandwidth_gb_s;       // 带宽利用率
    
    // 统计信息 (多次运行)
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float stddev_time_ms;
    
    int num_runs;               // 运行次数
    
    BenchmarkResult() : execution_time_ms(0.0f), gflops(0.0f), 
                       bandwidth_gb_s(0.0f), avg_time_ms(0.0f),
                       min_time_ms(0.0f), max_time_ms(0.0f),
                       stddev_time_ms(0.0f), num_runs(0) {}
};

// 基准测试配置
struct BenchmarkConfig {
    int num_warmup_runs;    // 预热运行次数
    int num_runs;           // 测试运行次数
    bool compare_cpu;       // 是否与 CPU 对比
    
    BenchmarkConfig() : num_warmup_runs(5), num_runs(20), compare_cpu(true) {}
};

// 运行 CSR SpMV 基准测试
BenchmarkResult benchmark_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config,
    const BenchmarkConfig* bench_config = nullptr
);

// 运行 ELL SpMV 基准测试
BenchmarkResult benchmark_ell(
    const ELLMatrix* A,
    const float* x,
    const BenchmarkConfig* bench_config = nullptr
);

// 对比 GPU vs CPU
struct ComparisonResult {
    BenchmarkResult gpu_result;
    BenchmarkResult cpu_result;
    float speedup;  // GPU speedup over CPU
    
    ComparisonResult() : speedup(0.0f) {}
};

ComparisonResult compare_gpu_cpu_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config,
    const BenchmarkConfig* bench_config = nullptr
);

// JSON 序列化
std::string benchmark_to_json(const BenchmarkResult& result);
std::string comparison_to_json(const ComparisonResult& result);

// JSON 反序列化
BenchmarkResult benchmark_from_json(const std::string& json);

} // namespace spmv

#endif // SPMV_BENCHMARK_H
