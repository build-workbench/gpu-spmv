#include "internal/csr_device.h"
#include "spmv/cuda_buffer.h"
#include "spmv/pagerank.h"
#include "spmv/spmv.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <vector>

namespace spmv {

__global__ void apply_pagerank_update_kernel(float* ranks, int n, float damping,
                                             float dangling_contrib, float teleport) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        ranks[idx] = damping * ranks[idx] + dangling_contrib + teleport;
    }
}

__global__ void accumulate_dangling_sum_kernel(const int* dangling_nodes, int num_dangling,
                                               const float* ranks, float* dangling_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_dangling) {
        atomicAdd(dangling_sum, ranks[dangling_nodes[idx]]);
    }
}

__global__ void compute_l2_diff_kernel(const float* a, const float* b, float* partial_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = a[idx] - b[idx];
        atomicAdd(partial_sums, diff * diff);
    }
}

static int map_cuda_exception_to_spmv_error(const CudaException& e) {
    return (e.error() == cudaErrorMemoryAllocation) ? static_cast<int>(SpMVError::CUDA_MALLOC)
                                                    : static_cast<int>(SpMVError::CUDA_MEMCPY);
}

static std::vector<int> find_dangling_nodes(const CSRMatrix* adj_matrix) {
    std::vector<int> dangling;
    if (!adj_matrix || adj_matrix->num_cols <= 0 || adj_matrix->num_rows <= 0) {
        return dangling;
    }
    if (!adj_matrix->values || !adj_matrix->col_indices || !adj_matrix->row_ptrs) {
        return dangling;
    }

    int num_cols = adj_matrix->num_cols;
    std::vector<float> col_sums(num_cols, 0.0f);
    for (int row = 0; row < adj_matrix->num_rows; row++) {
        int start = adj_matrix->row_ptrs[row];
        int end = adj_matrix->row_ptrs[row + 1];
        for (int idx = start; idx < end; idx++) {
            int col = adj_matrix->col_indices[idx];
            if (col >= 0 && col < num_cols) {
                col_sums[col] += adj_matrix->values[idx];
            }
        }
    }

    for (int col = 0; col < num_cols; col++) {
        if (col_sums[col] == 0.0f) {
            dangling.push_back(col);
        }
    }
    return dangling;
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

    if (!adj_matrix->row_ptrs || !csr_d_row_ptrs(adj_matrix) ||
        (adj_matrix->nnz > 0 && (!adj_matrix->values || !adj_matrix->col_indices ||
                                 !csr_d_values(adj_matrix) || !csr_d_col_indices(adj_matrix)))) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    PageRankConfig default_config;
    if (!config) {
        config = &default_config;
    }

    if (config->max_iterations < 0 || config->tolerance < 0.0f || config->damping_factor < 0.0f ||
        config->damping_factor > 1.0f) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int n = adj_matrix->num_rows;
    if (n == 0) {
        result.converged = true;
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    auto fail = [&result](int error_code) {
        if (result.ranks) {
            delete[] result.ranks;
            result.ranks = nullptr;
        }
        result.converged = false;
        result.error_code = error_code;
        return result;
    };

    try {
        result.ranks = new float[n];
        float init_rank = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; i++) {
            result.ranks[i] = init_rank;
        }

        CudaBuffer<float> d_ranks_old(n);
        CudaBuffer<float> d_ranks_new(n);
        CudaBuffer<float> d_scalar(1);

        d_ranks_old.copyFromHost(result.ranks, n);

        std::vector<int> dangling_nodes = find_dangling_nodes(adj_matrix);
        CudaBuffer<int> d_dangling_nodes(dangling_nodes.size());
        if (!dangling_nodes.empty()) {
            d_dangling_nodes.copyFromHost(dangling_nodes.data(), dangling_nodes.size());
        }

        float damping = config->damping_factor;
        float teleport = (1.0f - damping) / static_cast<float>(n);

        SpMVConfig spmv_config;
        spmv_config.kernel_type = SpMVConfig::VECTOR_CSR;
        SpMVExecutionContext context;

        const int block_size = 256;
        const int num_blocks = (n + block_size - 1) / block_size;
        const int dangling_blocks =
            dangling_nodes.empty()
                ? 0
                : static_cast<int>((dangling_nodes.size() + block_size - 1) / block_size);

        bool final_from_new = false;

        for (int iter = 0; iter < config->max_iterations; iter++) {
            d_scalar.memset();
            if (!dangling_nodes.empty()) {
                accumulate_dangling_sum_kernel<<<dangling_blocks, block_size>>>(
                    d_dangling_nodes.get(), static_cast<int>(dangling_nodes.size()),
                    d_ranks_old.get(), d_scalar.get());
                if (cudaGetLastError() != cudaSuccess) {
                    return fail(static_cast<int>(SpMVError::KERNEL_LAUNCH));
                }
            }

            float dangling_sum = 0.0f;
            d_scalar.copyToHost(&dangling_sum, 1);

            SpMVResult spmv_result = spmv_csr(adj_matrix, d_ranks_old.get(), d_ranks_new.get(),
                                              &spmv_config, n, &context);
            if (spmv_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
                return fail(spmv_result.error_code);
            }

            float dangling_contrib = damping * dangling_sum / static_cast<float>(n);
            apply_pagerank_update_kernel<<<num_blocks, block_size>>>(d_ranks_new.get(), n, damping,
                                                                     dangling_contrib, teleport);
            if (cudaGetLastError() != cudaSuccess) {
                return fail(static_cast<int>(SpMVError::KERNEL_LAUNCH));
            }

            d_scalar.memset();
            compute_l2_diff_kernel<<<num_blocks, block_size>>>(d_ranks_new.get(), d_ranks_old.get(),
                                                               d_scalar.get(), n);
            if (cudaGetLastError() != cudaSuccess) {
                return fail(static_cast<int>(SpMVError::KERNEL_LAUNCH));
            }

            float residual_sq = 0.0f;
            d_scalar.copyToHost(&residual_sq, 1);
            float residual = std::sqrt(residual_sq);

            result.iterations = iter + 1;
            result.final_residual = residual;

            if (residual < config->tolerance) {
                result.converged = true;
                final_from_new = true;
                break;
            }

            std::swap(d_ranks_old, d_ranks_new);
        }

        if (final_from_new) {
            d_ranks_new.copyToHost(result.ranks, n);
        } else {
            d_ranks_old.copyToHost(result.ranks, n);
        }

        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += result.ranks[i];
        }
        if (sum > 0.0f) {
            for (int i = 0; i < n; i++) {
                result.ranks[i] /= sum;
            }
        }

        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    } catch (const CudaException& e) {
        return fail(map_cuda_exception_to_spmv_error(e));
    } catch (const std::bad_alloc&) {
        return fail(static_cast<int>(SpMVError::OUT_OF_MEMORY));
    }
}

void pagerank_free(PageRankResult* result) {
    if (result && result->ranks) {
        delete[] result->ranks;
        result->ranks = nullptr;
    }
}

void pagerank_top_k(const PageRankResult* result, int num_nodes, int k, TopKNode* top_k) {
    if (!result || !result->ranks || !top_k || k <= 0 || num_nodes <= 0 ||
        result->error_code != static_cast<int>(SpMVError::SUCCESS)) {
        return;
    }

    std::vector<TopKNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        nodes[i].node_id = i;
        nodes[i].rank = result->ranks[i];
    }

    int actual_k = std::min(k, num_nodes);
    std::partial_sort(nodes.begin(), nodes.begin() + actual_k, nodes.end(),
                      [](const TopKNode& a, const TopKNode& b) { return a.rank > b.rank; });

    for (int i = 0; i < actual_k; i++) {
        top_k[i] = nodes[i];
    }
}

}  // namespace spmv
