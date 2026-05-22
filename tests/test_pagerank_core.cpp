#include "spmv/csr_matrix.h"
#include "spmv/pagerank.h"

#include <gtest/gtest.h>
#include <vector>

using namespace spmv;

#if !SPMV_WITH_CUDA

TEST(PageRankCoreTest, NoCudaBuildUsesWorkingBackendForSimpleCycle) {
    std::vector<float> adj = {0.0f, 0.0f, 1.0f,
                              1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    ASSERT_EQ(csr_from_dense(csr, adj.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));

    PageRankResult result = pagerank(csr, nullptr);

    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
    ASSERT_NE(result.ranks, nullptr);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.ranks[0], result.ranks[1], 1e-4f);
    EXPECT_NEAR(result.ranks[1], result.ranks[2], 1e-4f);

    pagerank_free(&result);
    csr_destroy(csr);
}

TEST(PageRankCoreTest, NoCudaBuildKeepsDanglingGraphNormalized) {
    std::vector<float> adj = {0.0f, 0.0f, 0.0f,
                              1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    ASSERT_EQ(csr_from_dense(csr, adj.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));

    PageRankConfig config;
    config.max_iterations = 100;
    config.tolerance = 1e-6f;

    PageRankResult result = pagerank(csr, &config);

    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
    ASSERT_NE(result.ranks, nullptr);

    float sum = result.ranks[0] + result.ranks[1] + result.ranks[2];
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
    EXPECT_GE(result.ranks[0], 0.0f);
    EXPECT_GE(result.ranks[1], 0.0f);
    EXPECT_GE(result.ranks[2], 0.0f);

    pagerank_free(&result);
    csr_destroy(csr);
}

#endif
