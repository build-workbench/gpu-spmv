#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>

#include "spmv/benchmark.h"
#include "spmv/cuda_buffer.h"

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

static int map_cuda_exception_to_spmv_error(const CudaException& e) {
  return (e.error() == cudaErrorMemoryAllocation)
             ? static_cast<int>(SpMVError::CUDA_MALLOC)
             : static_cast<int>(SpMVError::CUDA_MEMCPY);
}

static int validate_benchmark_config(const BenchmarkConfig* bench_config) {
  if (!bench_config) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (bench_config->num_warmup_runs < 0 || bench_config->num_runs <= 0) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  return static_cast<int>(SpMVError::SUCCESS);
}

static int validate_csr_device_benchmark_input(const CSRMatrix* A,
                                               const float* x) {
  if (!A || A->num_rows < 0 || A->num_cols < 0 || A->nnz < 0) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (A->num_cols > 0 && !x) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (!A->d_row_ptrs || (A->nnz > 0 && (!A->d_values || !A->d_col_indices))) {
    return static_cast<int>(SpMVError::INVALID_FORMAT);
  }
  return static_cast<int>(SpMVError::SUCCESS);
}

static int validate_ell_device_benchmark_input(const ELLMatrix* A,
                                               const float* x) {
  if (!A || A->num_rows < 0 || A->num_cols < 0 || A->max_nnz_per_row < 0 ||
      A->nnz < 0) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (A->num_cols > 0 && !x) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  size_t storage_size = static_cast<size_t>(A->num_rows) *
                        static_cast<size_t>(A->max_nnz_per_row);
  if (storage_size > 0 && (!A->d_values || !A->d_col_indices)) {
    return static_cast<int>(SpMVError::INVALID_FORMAT);
  }
  return static_cast<int>(SpMVError::SUCCESS);
}

static int validate_csr_host_benchmark_input(const CSRMatrix* A,
                                             const float* x) {
  if (!A || A->num_rows < 0 || A->num_cols < 0 || A->nnz < 0) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (A->num_cols > 0 && !x) {
    return static_cast<int>(SpMVError::INVALID_ARGUMENT);
  }
  if (!A->row_ptrs || (A->nnz > 0 && (!A->values || !A->col_indices))) {
    return static_cast<int>(SpMVError::INVALID_FORMAT);
  }
  return static_cast<int>(SpMVError::SUCCESS);
}

BenchmarkResult benchmark_csr(const CSRMatrix* A, const float* x,
                              const SpMVConfig* config,
                              const BenchmarkConfig* bench_config) {
  BenchmarkResult result;
  result.name = "CSR SpMV";

  BenchmarkConfig default_config;
  if (!bench_config) {
    bench_config = &default_config;
  }

  result.error_code = validate_benchmark_config(bench_config);
  if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    return result;
  }

  result.error_code = validate_csr_device_benchmark_input(A, x);
  if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    return result;
  }

  try {
    CudaBuffer<float> d_x(A->num_cols);
    CudaBuffer<float> d_y(A->num_rows);
    if (A->num_cols > 0) {
      d_x.copyFromHost(x, A->num_cols);
    }

    SpMVExecutionContext context;
    for (int i = 0; i < bench_config->num_warmup_runs; i++) {
      SpMVResult warmup_result =
          spmv_csr(A, d_x.get(), d_y.get(), config, A->num_cols, &context);
      if (warmup_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = warmup_result.error_code;
        return result;
      }
    }

    std::vector<float> times;
    times.reserve(bench_config->num_runs);

    for (int i = 0; i < bench_config->num_runs; i++) {
      SpMVResult spmv_result =
          spmv_csr(A, d_x.get(), d_y.get(), config, A->num_cols, &context);
      if (spmv_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
        result.num_runs = static_cast<int>(times.size());
        result.error_code = spmv_result.error_code;
        return result;
      }

      times.push_back(spmv_result.elapsed_ms);
      result.gflops = spmv_result.gflops;
      result.bandwidth_gb_s = spmv_result.bandwidth_gb_s;
    }

    result.num_runs = static_cast<int>(times.size());
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());

    float sum = 0.0f;
    for (float t : times) sum += t;
    result.avg_time_ms = sum / times.size();
    result.execution_time_ms = result.avg_time_ms;
    result.stddev_time_ms = compute_stddev(times, result.avg_time_ms);
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
  } catch (const CudaException& e) {
    result.error_code = map_cuda_exception_to_spmv_error(e);
    return result;
  } catch (const std::bad_alloc&) {
    result.error_code = static_cast<int>(SpMVError::OUT_OF_MEMORY);
    return result;
  }
}

BenchmarkResult benchmark_ell(const ELLMatrix* A, const float* x,
                              const BenchmarkConfig* bench_config) {
  BenchmarkResult result;
  result.name = "ELL SpMV";

  BenchmarkConfig default_config;
  if (!bench_config) {
    bench_config = &default_config;
  }

  result.error_code = validate_benchmark_config(bench_config);
  if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    return result;
  }

  result.error_code = validate_ell_device_benchmark_input(A, x);
  if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    return result;
  }

  try {
    CudaBuffer<float> d_x(A->num_cols);
    CudaBuffer<float> d_y(A->num_rows);
    if (A->num_cols > 0) {
      d_x.copyFromHost(x, A->num_cols);
    }

    SpMVExecutionContext context;
    for (int i = 0; i < bench_config->num_warmup_runs; i++) {
      SpMVResult warmup_result =
          spmv_ell(A, d_x.get(), d_y.get(), nullptr, A->num_cols, &context);
      if (warmup_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = warmup_result.error_code;
        return result;
      }
    }

    std::vector<float> times;
    times.reserve(bench_config->num_runs);

    for (int i = 0; i < bench_config->num_runs; i++) {
      SpMVResult spmv_result =
          spmv_ell(A, d_x.get(), d_y.get(), nullptr, A->num_cols, &context);
      if (spmv_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
        result.num_runs = static_cast<int>(times.size());
        result.error_code = spmv_result.error_code;
        return result;
      }

      times.push_back(spmv_result.elapsed_ms);
      result.gflops = spmv_result.gflops;
      result.bandwidth_gb_s = spmv_result.bandwidth_gb_s;
    }

    result.num_runs = static_cast<int>(times.size());
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());

    float sum = 0.0f;
    for (float t : times) sum += t;
    result.avg_time_ms = sum / times.size();
    result.execution_time_ms = result.avg_time_ms;
    result.stddev_time_ms = compute_stddev(times, result.avg_time_ms);
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
  } catch (const CudaException& e) {
    result.error_code = map_cuda_exception_to_spmv_error(e);
    return result;
  } catch (const std::bad_alloc&) {
    result.error_code = static_cast<int>(SpMVError::OUT_OF_MEMORY);
    return result;
  }
}

ComparisonResult compare_gpu_cpu_csr(const CSRMatrix* A, const float* x,
                                     const SpMVConfig* config,
                                     const BenchmarkConfig* bench_config) {
  ComparisonResult comp;

  BenchmarkConfig default_config;
  if (!bench_config) {
    bench_config = &default_config;
  }

  int config_status = validate_benchmark_config(bench_config);
  if (config_status != static_cast<int>(SpMVError::SUCCESS)) {
    comp.gpu_result.error_code = config_status;
    comp.cpu_result.error_code = config_status;
    comp.error_code = config_status;
    return comp;
  }

  int host_status = validate_csr_host_benchmark_input(A, x);
  if (host_status != static_cast<int>(SpMVError::SUCCESS)) {
    comp.gpu_result.error_code = host_status;
    comp.cpu_result.error_code = host_status;
    comp.error_code = host_status;
    return comp;
  }

  comp.gpu_result = benchmark_csr(A, x, config, bench_config);
  if (comp.gpu_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    comp.cpu_result.error_code = comp.gpu_result.error_code;
    comp.error_code = comp.gpu_result.error_code;
    return comp;
  }

  comp.cpu_result.name = "CPU CSR SpMV";

  try {
    std::vector<float> y(A->num_rows);
    std::vector<float> times;
    times.reserve(bench_config->num_runs);

    for (int i = 0; i < bench_config->num_runs; i++) {
      auto t0 = std::chrono::high_resolution_clock::now();
      spmv_cpu_csr(A, x, y.data());
      auto t1 = std::chrono::high_resolution_clock::now();

      float elapsed_ms =
          std::chrono::duration<float, std::milli>(t1 - t0).count();
      times.push_back(elapsed_ms);
    }

    for (float& t : times) {
      if (t <= 0.0f) {
        t = std::numeric_limits<float>::epsilon();
      }
    }

    comp.cpu_result.num_runs = static_cast<int>(times.size());
    comp.cpu_result.min_time_ms = *std::min_element(times.begin(), times.end());
    comp.cpu_result.max_time_ms = *std::max_element(times.begin(), times.end());

    float sum = 0.0f;
    for (float t : times) sum += t;
    comp.cpu_result.avg_time_ms = sum / times.size();
    comp.cpu_result.execution_time_ms = comp.cpu_result.avg_time_ms;
    comp.cpu_result.stddev_time_ms =
        compute_stddev(times, comp.cpu_result.avg_time_ms);
    comp.cpu_result.error_code = static_cast<int>(SpMVError::SUCCESS);
    comp.error_code = static_cast<int>(SpMVError::SUCCESS);

    if (comp.gpu_result.avg_time_ms > 0.0f) {
      comp.speedup = comp.cpu_result.avg_time_ms / comp.gpu_result.avg_time_ms;
    }

    return comp;
  } catch (const std::bad_alloc&) {
    comp.cpu_result.error_code = static_cast<int>(SpMVError::OUT_OF_MEMORY);
    comp.error_code = comp.cpu_result.error_code;
    return comp;
  }
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
  oss << "  \"num_runs\": " << result.num_runs << ",\n";
  oss << "  \"error_code\": " << result.error_code << "\n";
  oss << "}";
  return oss.str();
}

std::string comparison_to_json(const ComparisonResult& result) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(6);
  oss << "{\n";
  oss << "  \"gpu\": " << benchmark_to_json(result.gpu_result) << ",\n";
  oss << "  \"cpu\": " << benchmark_to_json(result.cpu_result) << ",\n";
  oss << "  \"speedup\": " << result.speedup << ",\n";
  oss << "  \"error_code\": " << result.error_code << "\n";
  oss << "}";
  return oss.str();
}

BenchmarkResult benchmark_from_json(const std::string& json) {
  // 简单的 JSON 解析 (仅用于测试)
  BenchmarkResult result;

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
  result.error_code = static_cast<int>(find_value("error_code"));

  return result;
}

}  // namespace spmv
