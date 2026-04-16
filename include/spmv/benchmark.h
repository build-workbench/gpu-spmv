#ifndef SPMV_BENCHMARK_H
#define SPMV_BENCHMARK_H

#include <string>
#include <vector>

#include "csr_matrix.h"
#include "ell_matrix.h"
#include "spmv.h"

namespace spmv {

/**
 * @file benchmark.h
 * @brief Benchmarking utilities for SpMV operations.
 *
 * Provides tools for measuring SpMV performance with
 * multiple runs, statistics, and GPU/CPU comparison.
 */

/**
 * @brief Result of a benchmark run.
 *
 * Contains timing statistics from multiple runs.
 */
struct BenchmarkResult {
    std::string name;         ///< Test name
    float execution_time_ms;  ///< Execution time (ms)
    float gflops;             ///< Computed GFLOPS
    float bandwidth_gb_s;     ///< Memory bandwidth (GB/s)

    // Statistics from multiple runs
    float avg_time_ms;     ///< Average time across runs
    float min_time_ms;     ///< Minimum time
    float max_time_ms;     ///< Maximum time
    float stddev_time_ms;  ///< Standard deviation

    int num_runs;    ///< Number of successful runs
    int error_code;  ///< 0 = success, negative = error

    BenchmarkResult()
        : execution_time_ms(0.0f),
          gflops(0.0f),
          bandwidth_gb_s(0.0f),
          avg_time_ms(0.0f),
          min_time_ms(0.0f),
          max_time_ms(0.0f),
          stddev_time_ms(0.0f),
          num_runs(0),
          error_code(static_cast<int>(SpMVError::SUCCESS)) {}
};

/**
 * @brief Configuration for benchmark runs.
 */
struct BenchmarkConfig {
    int num_warmup_runs;  ///< Warmup runs (not timed)
    int num_runs;         ///< Timed runs
    bool compare_cpu;     ///< Include CPU comparison

    BenchmarkConfig() : num_warmup_runs(5), num_runs(20), compare_cpu(true) {}
};

/**
 * @brief Run CSR SpMV benchmark.
 *
 * @param A CSR matrix with device data.
 * @param x Input vector (device memory).
 * @param config SpMV kernel configuration.
 * @param bench_config Benchmark settings.
 * @return Benchmark results.
 */
BenchmarkResult benchmark_csr(const CSRMatrix* A, const float* x, const SpMVConfig* config,
                              const BenchmarkConfig* bench_config = nullptr);

/**
 * @brief Run ELL SpMV benchmark.
 *
 * @param A ELL matrix with device data.
 * @param x Input vector (device memory).
 * @param bench_config Benchmark settings.
 * @return Benchmark results.
 */
BenchmarkResult benchmark_ell(const ELLMatrix* A, const float* x,
                              const BenchmarkConfig* bench_config = nullptr);

/**
 * @brief Result of GPU vs CPU comparison.
 */
struct ComparisonResult {
    BenchmarkResult gpu_result;  ///< GPU benchmark result
    BenchmarkResult cpu_result;  ///< CPU benchmark result
    float speedup;               ///< GPU speedup factor
    int error_code;              ///< 0 = success

    ComparisonResult() : speedup(0.0f), error_code(static_cast<int>(SpMVError::SUCCESS)) {}
};

/**
 * @brief Compare GPU and CPU SpMV performance.
 *
 * @param A CSR matrix with device data.
 * @param x Input vector.
 * @param config SpMV configuration.
 * @param bench_config Benchmark settings.
 * @return Comparison results.
 */
ComparisonResult compare_gpu_cpu_csr(const CSRMatrix* A, const float* x, const SpMVConfig* config,
                                     const BenchmarkConfig* bench_config = nullptr);

/**
 * @brief Serialize benchmark result to JSON.
 * @param result Benchmark result.
 * @return JSON string.
 */
std::string benchmark_to_json(const BenchmarkResult& result);

/**
 * @brief Serialize comparison result to JSON.
 * @param result Comparison result.
 * @return JSON string.
 */
std::string comparison_to_json(const ComparisonResult& result);

/**
 * @brief Parse benchmark result from JSON.
 * @param json JSON string.
 * @return Benchmark result.
 */
BenchmarkResult benchmark_from_json(const std::string& json);

}  // namespace spmv

#endif  // SPMV_BENCHMARK_H
