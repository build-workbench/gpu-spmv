#include "spmv/benchmark.h"
#include "spmv/pagerank.h"
#include "spmv/spmv.h"
#include "internal/pagerank_common.h"

#include <cmath>
#include <new>
#include <sstream>
#include <vector>

namespace spmv {

namespace {

int no_cuda_error() {
    return static_cast<int>(SpMVError::KERNEL_LAUNCH);
}

}  // namespace

SpMVResult spmv_csr(const CSRMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    SpMVExecutionContext*) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = no_cuda_error();
    return result;
}

SpMVResult spmv_ell(const ELLMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    SpMVExecutionContext*) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = no_cuda_error();
    return result;
}

BenchmarkResult benchmark_csr(const CSRMatrix*, const float*, const SpMVConfig*,
                              const BenchmarkConfig*) {
    BenchmarkResult result;
    result.error_code = no_cuda_error();
    return result;
}

BenchmarkResult benchmark_ell(const ELLMatrix*, const float*, const BenchmarkConfig*) {
    BenchmarkResult result;
    result.error_code = no_cuda_error();
    return result;
}

ComparisonResult compare_gpu_cpu_csr(const CSRMatrix*, const float*, const SpMVConfig*,
                                     const BenchmarkConfig*) {
    ComparisonResult result;
    result.error_code = no_cuda_error();
    result.gpu_result.error_code = no_cuda_error();
    result.cpu_result.error_code = no_cuda_error();
    return result;
}

std::string benchmark_to_json(const BenchmarkResult& result) {
    std::ostringstream json;
    json << "{\"name\":\"" << result.name << "\",\"execution_time_ms\":" << result.execution_time_ms
         << ",\"gflops\":" << result.gflops << ",\"bandwidth_gb_s\":" << result.bandwidth_gb_s
         << ",\"avg_time_ms\":" << result.avg_time_ms << ",\"min_time_ms\":" << result.min_time_ms
         << ",\"max_time_ms\":" << result.max_time_ms << ",\"stddev_time_ms\":"
         << result.stddev_time_ms << ",\"num_runs\":" << result.num_runs << ",\"error_code\":"
         << result.error_code << "}";
    return json.str();
}

std::string comparison_to_json(const ComparisonResult& result) {
    std::ostringstream json;
    json << "{\"speedup\":" << result.speedup << ",\"error_code\":" << result.error_code << "}";
    return json.str();
}

BenchmarkResult benchmark_from_json(const std::string&) {
    BenchmarkResult result;
    result.error_code = no_cuda_error();
    return result;
}

PageRankResult pagerank(const CSRMatrix* adj_matrix, const PageRankConfig* config) {
    PageRankResult result;

    if (!adj_matrix) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }
    if (adj_matrix->num_rows < 0 || adj_matrix->num_cols < 0 || adj_matrix->nnz < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }
    if (adj_matrix->num_rows != adj_matrix->num_cols) {
        result.error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return result;
    }
    if (!adj_matrix->row_ptrs ||
        (adj_matrix->nnz > 0 && (!adj_matrix->values || !adj_matrix->col_indices))) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    PageRankConfig default_config;
    if (!config) {
        config = &default_config;
    }
    if (config->max_iterations < 0 || config->tolerance < 0.0f ||
        config->damping_factor < 0.0f || config->damping_factor > 1.0f) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int n = adj_matrix->num_rows;
    if (n == 0) {
        result.converged = true;
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    result.ranks = new (std::nothrow) float[n];
    if (!result.ranks) {
        result.error_code = static_cast<int>(SpMVError::OUT_OF_MEMORY);
        return result;
    }

    float init_rank = 1.0f / static_cast<float>(n);
    std::vector<float> next_ranks(n, 0.0f);
    for (int i = 0; i < n; i++) {
        result.ranks[i] = init_rank;
    }

    std::vector<int> dangling_nodes = pagerank_find_dangling_nodes(adj_matrix);
    float damping = config->damping_factor;
    float teleport = (1.0f - damping) / static_cast<float>(n);

    for (int iter = 0; iter < config->max_iterations; iter++) {
        float dangling_sum = 0.0f;
        for (int node : dangling_nodes) {
            dangling_sum += result.ranks[node];
        }

        spmv_cpu_csr(adj_matrix, result.ranks, next_ranks.data());

        float dangling_contrib = damping * dangling_sum / static_cast<float>(n);
        float residual_sq = 0.0f;
        for (int i = 0; i < n; i++) {
            next_ranks[i] = damping * next_ranks[i] + dangling_contrib + teleport;
            float diff = next_ranks[i] - result.ranks[i];
            residual_sq += diff * diff;
        }

        result.iterations = iter + 1;
        result.final_residual = std::sqrt(residual_sq);

        for (int i = 0; i < n; i++) {
            result.ranks[i] = next_ranks[i];
        }

        if (result.final_residual < config->tolerance) {
            result.converged = true;
            break;
        }
    }

    pagerank_normalize(result.ranks, n);
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
    return result;
}

}  // namespace spmv
