#ifndef SPMV_PAGERANK_H
#define SPMV_PAGERANK_H

#include "csr_matrix.h"

namespace spmv {

// PageRank 配置
struct PageRankConfig {
    float damping_factor;    // 阻尼系数，通常 0.85
    float tolerance;         // 收敛阈值，默认 1e-6
    int max_iterations;      // 最大迭代次数
    
    PageRankConfig() : damping_factor(0.85f), tolerance(1e-6f), max_iterations(100) {}
};

// PageRank 结果
struct PageRankResult {
    float* ranks;            // 排名分数 [num_nodes]
    int iterations;          // 实际迭代次数
    float final_residual;    // 最终残差
    bool converged;          // 是否收敛
    
    PageRankResult() : ranks(nullptr), iterations(0), final_residual(0.0f), converged(false) {}
};

// PageRank 算法
// adj_matrix: 列归一化的邻接矩阵 (CSR 格式)
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,
    const PageRankConfig* config = nullptr
);

// 释放 PageRank 结果
void pagerank_free(PageRankResult* result);

// 获取 Top-K 节点
struct TopKNode {
    int node_id;
    float rank;
};

void pagerank_top_k(const PageRankResult* result, int num_nodes, int k, TopKNode* top_k);

} // namespace spmv

#endif // SPMV_PAGERANK_H
