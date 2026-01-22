#include "spmv/pagerank.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace spmv {

// 计算 L2 范数
static float compute_l2_norm(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
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

PageRankResult pagerank(
    const CSRMatrix* adj_matrix,
    const PageRankConfig* config
) {
    PageRankResult result;
    
    if (!adj_matrix) {
        return result;
    }
    
    PageRankConfig default_config;
    if (!config) {
        config = &default_config;
    }
    
    int n = adj_matrix->num_rows;
    
    // 初始化排名分数
    result.ranks = new float[n];
    float init_rank = 1.0f / n;
    for (int i = 0; i < n; i++) {
        result.ranks[i] = init_rank;
    }
    
    std::vector<float> ranks_old(n);
    std::vector<float> ranks_new(n);
    std::copy(result.ranks, result.ranks + n, ranks_old.begin());
    
    // 分配 GPU 内存
    CudaBuffer<float> d_ranks_old(n);
    CudaBuffer<float> d_ranks_new(n);
    
    d_ranks_old.copyFromHost(result.ranks, n);
    
    // PageRank 迭代
    float damping = config->damping_factor;
    float teleport = (1.0f - damping) / n;
    std::vector<int> dangling_nodes = find_dangling_nodes(adj_matrix);
    
    SpMVConfig spmv_config;
    spmv_config.kernel_type = SpMVConfig::VECTOR_CSR;
    bool final_from_new = false;
    
    for (int iter = 0; iter < config->max_iterations; iter++) {
        float dangling_sum = 0.0f;
        for (int node : dangling_nodes) {
            if (node >= 0 && node < n) {
                dangling_sum += ranks_old[node];
            }
        }

        // r_new = d * (A * r_old + dangling_sum / n) + (1-d) / n
        SpMVResult spmv_result = spmv_csr(adj_matrix, d_ranks_old.get(), 
                                          d_ranks_new.get(), &spmv_config, n);
        
        if (spmv_result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
            break;
        }
        
        // 添加 teleport 项
        d_ranks_new.copyToHost(ranks_new.data(), n);
        float dangling_contrib = (n > 0) ? (damping * dangling_sum / n) : 0.0f;
        for (int i = 0; i < n; i++) {
            ranks_new[i] = damping * ranks_new[i] + dangling_contrib + teleport;
        }
        d_ranks_new.copyFromHost(ranks_new.data(), n);
        
        // 检查收敛
        float residual = compute_l2_norm(ranks_new.data(), ranks_old.data(), n);
        
        result.iterations = iter + 1;
        result.final_residual = residual;
        
        if (residual < config->tolerance) {
            result.converged = true;
            final_from_new = true;
            break;
        }
        
        // 交换
        std::swap(d_ranks_old, d_ranks_new);
        std::swap(ranks_old, ranks_new);
    }
    
    // 复制最终结果
    if (final_from_new) {
        d_ranks_new.copyToHost(result.ranks, n);
    } else {
        d_ranks_old.copyToHost(result.ranks, n);
    }
    
    // 归一化确保和为 1
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += result.ranks[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) {
            result.ranks[i] /= sum;
        }
    }
    
    return result;
}

void pagerank_free(PageRankResult* result) {
    if (result && result->ranks) {
        delete[] result->ranks;
        result->ranks = nullptr;
    }
}

void pagerank_top_k(const PageRankResult* result, int num_nodes, int k, TopKNode* top_k) {
    if (!result || !result->ranks || !top_k || k <= 0) {
        return;
    }
    
    // 创建节点-排名对
    std::vector<TopKNode> nodes(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        nodes[i].node_id = i;
        nodes[i].rank = result->ranks[i];
    }
    
    // 部分排序获取 Top-K
    int actual_k = std::min(k, num_nodes);
    std::partial_sort(nodes.begin(), nodes.begin() + actual_k, nodes.end(),
                     [](const TopKNode& a, const TopKNode& b) {
                         return a.rank > b.rank;  // 降序
                     });
    
    // 复制结果
    for (int i = 0; i < actual_k; i++) {
        top_k[i] = nodes[i];
    }
}

} // namespace spmv
