#include <gtest/gtest.h>
#include "spmv/pagerank.h"
#include "spmv/csr_matrix.h"
#include "spmv/test_utils.h"
#include <cmath>

using namespace spmv;
using namespace spmv::test;

class PageRankPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 50;
};

// **Feature: spmv-gpu, Property 15: PageRank Score Invariants**
// **Validates: Requirements 7.1, 7.2**
TEST_F(PageRankPropertyTest, ScoreInvariants) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int n = rng.randInt(5, 50);
        float density = rng.randFloat(0.1f, 0.5f);
        
        // 生成随机邻接矩阵
        auto adj = generateRandomDenseMatrix(n, n, density, rng, 0.0f, 1.0f);
        
        // 列归一化
        for (int j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                col_sum += adj[i * n + j];
            }
            if (col_sum > 0.0f) {
                for (int i = 0; i < n; i++) {
                    adj[i * n + j] /= col_sum;
                }
            }
        }
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, adj.data(), n, n);
        csr_to_gpu(csr);
        
        PageRankConfig config;
        config.max_iterations = 50;
        config.tolerance = 1e-5f;
        
        PageRankResult result = pagerank(csr, &config);
        
        // 验证不变量
        // 1. 所有分数非负
        for (int i = 0; i < n; i++) {
            EXPECT_GE(result.ranks[i], 0.0f)
                << "Rank should be non-negative at node " << i 
                << " iteration " << iter;
        }
        
        // 2. 分数和为 1
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += result.ranks[i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f)
            << "Ranks should sum to 1.0 at iteration " << iter;
        
        // 3. 收敛或达到最大迭代次数
        EXPECT_TRUE(result.converged || result.iterations == config.max_iterations)
            << "Should converge or reach max iterations at iteration " << iter;
        
        if (result.converged) {
            EXPECT_LT(result.final_residual, config.tolerance)
                << "Converged residual should be below tolerance at iteration " << iter;
        }
        
        pagerank_free(&result);
        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 16: PageRank Top-K Ordering**
// **Validates: Requirements 7.5**
TEST_F(PageRankPropertyTest, TopKOrdering) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int n = rng.randInt(10, 50);
        int k = rng.randInt(3, std::min(10, n));
        
        auto adj = generateRandomDenseMatrix(n, n, 0.2f, rng, 0.0f, 1.0f);
        
        // 列归一化
        for (int j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                col_sum += adj[i * n + j];
            }
            if (col_sum > 0.0f) {
                for (int i = 0; i < n; i++) {
                    adj[i * n + j] /= col_sum;
                }
            }
        }
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, adj.data(), n, n);
        csr_to_gpu(csr);
        
        PageRankResult result = pagerank(csr, nullptr);
        
        std::vector<TopKNode> top_k(k);
        pagerank_top_k(&result, n, k, top_k.data());
        
        // 验证 Top-K 降序排列
        for (int i = 0; i < k - 1; i++) {
            EXPECT_GE(top_k[i].rank, top_k[i + 1].rank)
                << "Top-K should be in descending order at position " << i
                << " iteration " << iter;
        }
        
        // 验证 Top-K 中的节点排名高于其他节点
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                bool in_top_k = false;
                for (int m = 0; m < k; m++) {
                    if (top_k[m].node_id == j) {
                        in_top_k = true;
                        break;
                    }
                }
                if (!in_top_k) {
                    EXPECT_GE(top_k[i].rank, result.ranks[j])
                        << "Top-K node should have higher rank than non-top-k nodes";
                }
            }
        }
        
        pagerank_free(&result);
        csr_destroy(csr);
    }
}

// 单元测试
TEST(PageRankUnitTest, SimpleGraph) {
    // 简单的 3 节点图
    // 0 -> 1, 1 -> 2, 2 -> 0
    std::vector<float> adj = {
        0, 0, 1,
        1, 0, 0,
        0, 1, 0
    };
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, adj.data(), 3, 3);
    csr_to_gpu(csr);
    
    PageRankResult result = pagerank(csr, nullptr);
    
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0);
    
    // 对称图应该有相等的排名
    EXPECT_NEAR(result.ranks[0], result.ranks[1], 1e-4f);
    EXPECT_NEAR(result.ranks[1], result.ranks[2], 1e-4f);
    
    pagerank_free(&result);
    csr_destroy(csr);
}

TEST(PageRankUnitTest, TopKExtraction) {
    std::vector<float> adj = {
        0, 0.5f, 0.5f, 0,
        0.5f, 0, 0, 0.5f,
        0.5f, 0, 0, 0.5f,
        0, 0.5f, 0.5f, 0
    };
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, adj.data(), 4, 4);
    csr_to_gpu(csr);
    
    PageRankResult result = pagerank(csr, nullptr);
    
    std::vector<TopKNode> top_2(2);
    pagerank_top_k(&result, 4, 2, top_2.data());
    
    EXPECT_GE(top_2[0].rank, top_2[1].rank);
    EXPECT_GE(top_2[0].node_id, 0);
    EXPECT_LT(top_2[0].node_id, 4);
    
    pagerank_free(&result);
    csr_destroy(csr);
}
