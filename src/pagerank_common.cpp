#include "internal/pagerank_common.h"

#include <algorithm>

namespace spmv {

std::vector<int> pagerank_find_dangling_nodes(const CSRMatrix* adj_matrix) {
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

void pagerank_normalize(float* ranks, int n) {
    if (!ranks || n <= 0) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += ranks[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    for (int i = 0; i < n; i++) {
        ranks[i] /= sum;
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
