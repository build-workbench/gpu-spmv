#ifndef SPMV_PAGERANK_H
#define SPMV_PAGERANK_H

#include "csr_matrix.h"

namespace spmv {

/**
 * @file pagerank.h
 * @brief PageRank algorithm implementation using SpMV.
 *
 * Implements the PageRank algorithm for ranking nodes in a graph.
 * Uses iterative sparse matrix-vector multiplication.
 */

/**
 * @brief Configuration for PageRank algorithm.
 */
struct PageRankConfig {
  float damping_factor;  ///< Damping factor (typically 0.85)
  float tolerance;       ///< Convergence threshold (default 1e-6)
  int max_iterations;    ///< Maximum iterations

  PageRankConfig()
      : damping_factor(0.85f), tolerance(1e-6f), max_iterations(100) {}
};

/**
 * @brief Result of PageRank computation.
 */
struct PageRankResult {
  float* ranks;          ///< PageRank scores [num_nodes]
  int iterations;        ///< Actual iterations performed
  float final_residual;  ///< Final residual value
  bool converged;        ///< Whether algorithm converged
  int error_code;        ///< 0 = success, negative = error

  PageRankResult()
      : ranks(nullptr),
        iterations(0),
        final_residual(0.0f),
        converged(false),
        error_code(static_cast<int>(SpMVError::SUCCESS)) {}
};

/**
 * @brief Compute PageRank for a graph.
 *
 * The input matrix should be a column-normalized adjacency matrix
 * in CSR format. Each column should sum to 1.0 (or be all zeros
 * for dangling nodes).
 *
 * @param adj_matrix Column-normalized adjacency matrix (CSR format).
 * @param config Algorithm configuration (nullptr = defaults).
 * @return PageRank result with scores.
 */
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config = nullptr);

/**
 * @brief Free PageRank result memory.
 *
 * @param result Result to free.
 */
void pagerank_free(PageRankResult* result);

/**
 * @brief Node with its PageRank score for top-K queries.
 */
struct TopKNode {
  int node_id;  ///< Node identifier
  float rank;   ///< PageRank score
};

/**
 * @brief Get top-K nodes by PageRank score.
 *
 * @param result PageRank result.
 * @param num_nodes Total number of nodes.
 * @param k Number of top nodes to retrieve.
 * @param top_k Output array of TopKNode [k].
 */
void pagerank_top_k(const PageRankResult* result, int num_nodes, int k,
                    TopKNode* top_k);

}  // namespace spmv

#endif  // SPMV_PAGERANK_H
