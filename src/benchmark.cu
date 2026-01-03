#include "spmv/benchmark.h"
#include "spmv/cuda_buffer.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace spmv {

static float compute_stddev(const std::vector<float>& values, float mean) {
    if (values.size() <= 1) return 0.0f;
    
    float sum_sq_diff = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (values.size() - 1));
}

BenchmarkResult benchmark_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config,
    const BenchmarkConfig* bench_config
) {
    BenchmarkResult result;
    result.name = "CSR SpMV";
    
    BenchmarkConfig default_config;
    if (!bench_config) {
        bench_config = &default_config;
    }
    
    // 分配 GPU 内存
    CudaBuffer<float> d_x(A->num_cols);
    CudaBuffer<float> d_y(A->num_rows);
    d_x.copyFromHost(x, A->num_cols);
    
    // 预热
    for (int i = 0; i < bench_config->num_warmup_runs; i++) {
        spmv_csr(A, d_x.get(), d_y.get(), config);
    }
    
    // 测试运行
    std::vector<float> times;
    times.reserve(bench_config->num_runs);
    
    for (int i = 0; i < bench_config->num_runs; i++) {
        SpMVResult spmv_result = spmv_csr(A, d_x.get(), d_y.get(), config);
        if (spmv_result.error_code == static_cast<int>(SpMVError::SUCCESS)) {
            times.push_back(spmv_result.elapsed_ms);
            result.gflops = spmv_result.gflops;
            result.bandwidth_gb_s = spmv_result.bandwidth_gb_s;
        }
    }
    
    if (times.empty()) {
        return result;
    }
    
    // 计算统计信息
    result.num_runs = times.size();
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());
    
    float sum = 0.0f;
    for (float t : times) sum += t;
    result.avg_time_ms = sum / times.size();
    result.execution_time_ms = result.avg_time_ms;
    
    result.stddev_time_ms = compute_stddev(times, result.avg_time_ms);
    
    return result;
}

BenchmarkResult benchmark_ell(
    const ELLMatrix* A,
    const float* x,
    const BenchmarkConfig* bench_config
) {
    BenchmarkResult result;
    result.name = "ELL SpMV";
    
    BenchmarkConfig default_config;
    if (!bench_config) {
        bench_config = &default_config;
    }
    
    CudaBuffer<float> d_x(A->num_cols);
    CudaBuffer<float> d_y(A->num_rows);
    d_x.copyFromHost(x, A->num_cols);
    
    for (int i = 0; i < bench_config->num_warmup_runs; i++) {
        spmv_ell(A, d_x.get(), d_y.get(), nullptr);
    }
    
    std::vector<float> times;
    times.reserve(bench_config->num_runs);
    
    for (int i = 0; i < bench_config->num_runs; i++) {
        SpMVResult spmv_result = spmv_ell(A, d_x.get(), d_y.get(), nullptr);
        if (spmv_result.error_code == static_cast<int>(SpMVError::SUCCESS)) {
            times.push_back(spmv_result.elapsed_ms);
            result.gflops = spmv_result.gflops;
            result.bandwidth_gb_s = spmv_result.bandwidth_gb_s;
        }
    }
    
    if (times.empty()) {
        return result;
    }
    
    result.num_runs = times.size();
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());
    
    float sum = 0.0f;
    for (float t : times) sum += t;
    result.avg_time_ms = sum / times.size();
    result.execution_time_ms = result.avg_time_ms;
    
    result.stddev_time_ms = compute_stddev(times, result.avg_time_ms);
    
    return result;
}

ComparisonResult compare_gpu_cpu_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config,
    const BenchmarkConfig* bench_config
) {
    ComparisonResult comp;
    
    // GPU 基准测试
    comp.gpu_result = benchmark_csr(A, x, config, bench_config);
    
    // CPU 基准测试
    BenchmarkConfig default_config;
    if (!bench_config) {
        bench_config = &default_config;
    }
    
    comp.cpu_result.name = "CPU CSR SpMV";
    std::vector<float> y(A->num_rows);
    
    std::vector<float> times;
    times.reserve(bench_config->num_runs);
    
    for (int i = 0; i < bench_config->num_runs; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        spmv_cpu_csr(A, x, y.data());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times.push_back(elapsed_ms);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    comp.cpu_result.num_runs = times.size();
    comp.cpu_result.min_time_ms = *std::min_element(times.begin(), times.end());
    comp.cpu_result.max_time_ms = *std::max_element(times.begin(), times.end());
    
    float sum = 0.0f;
    for (float t : times) sum += t;
    comp.cpu_result.avg_time_ms = sum / times.size();
    comp.cpu_result.execution_time_ms = comp.cpu_result.avg_time_ms;
    comp.cpu_result.stddev_time_ms = compute_stddev(times, comp.cpu_result.avg_time_ms);
    
    // 计算加速比
    if (comp.gpu_result.avg_time_ms > 0.0f) {
        comp.speedup = comp.cpu_result.avg_time_ms / comp.gpu_result.avg_time_ms;
    }
    
    return comp;
}

std::string benchmark_to_json(const BenchmarkResult& result) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\n";
    oss << "  \"name\": \"" << result.name << "\",\n";
    oss << "  \"execution_time_ms\": " << result.execution_time_ms << ",\n";
    oss << "  \"gflops\": " << result.gflops << ",\n";
    oss << "  \"bandwidth_gb_s\": " << result.bandwidth_gb_s << ",\n";
    oss << "  \"avg_time_ms\": " << result.avg_time_ms << ",\n";
    oss << "  \"min_time_ms\": " << result.min_time_ms << ",\n";
    oss << "  \"max_time_ms\": " << result.max_time_ms << ",\n";
    oss << "  \"stddev_time_ms\": " << result.stddev_time_ms << ",\n";
    oss << "  \"num_runs\": " << result.num_runs << "\n";
    oss << "}";
    return oss.str();
}

std::string comparison_to_json(const ComparisonResult& result) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\n";
    oss << "  \"gpu\": " << benchmark_to_json(result.gpu_result) << ",\n";
    oss << "  \"cpu\": " << benchmark_to_json(result.cpu_result) << ",\n";
    oss << "  \"speedup\": " << result.speedup << "\n";
    oss << "}";
    return oss.str();
}

BenchmarkResult benchmark_from_json(const std::string& json) {
    // 简单的 JSON 解析 (仅用于测试)
    BenchmarkResult result;
    
    // 查找各个字段
    auto find_value = [&json](const std::string& key) -> float {
        size_t pos = json.find("\"" + key + "\":");
        if (pos == std::string::npos) return 0.0f;
        pos = json.find(":", pos) + 1;
        return std::stof(json.substr(pos));
    };
    
    result.execution_time_ms = find_value("execution_time_ms");
    result.gflops = find_value("gflops");
    result.bandwidth_gb_s = find_value("bandwidth_gb_s");
    result.avg_time_ms = find_value("avg_time_ms");
    result.min_time_ms = find_value("min_time_ms");
    result.max_time_ms = find_value("max_time_ms");
    result.stddev_time_ms = find_value("stddev_time_ms");
    result.num_runs = static_cast<int>(find_value("num_runs"));
    
    return result;
}

} // namespace spmv
