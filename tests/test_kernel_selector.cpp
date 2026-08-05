#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"

#include <gtest/gtest.h>

#include "test_utils.h"

using namespace spmv;
using namespace spmv::test;

class KernelSelectorPropertyTest : public ::testing::Test {
   protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

TEST_F(KernelSelectorPropertyTest, SelectorValidity) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 500);
        int cols = rng.randInt(1, 500);
        float density = rng.randFloat(0.001f, 0.5f);

        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);

        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);

        SpMVConfig config = spmv_auto_config(csr);

        EXPECT_GE(config.block_size, 32) << "Block size too small at iteration " << iter;
        EXPECT_LE(config.block_size, 1024) << "Block size too large at iteration " << iter;
        EXPECT_EQ(config.block_size % 32, 0)
            << "Block size not multiple of 32 at iteration " << iter;

        EXPECT_TRUE(config.kernel_type == SpMVConfig::SCALAR_CSR ||
                    config.kernel_type == SpMVConfig::VECTOR_CSR ||
                    config.kernel_type == SpMVConfig::MERGE_PATH ||
                    config.kernel_type == SpMVConfig::ELL_KERNEL)
            << "Invalid kernel type at iteration " << iter;

        csr_destroy(csr);
    }
}

TEST(KernelSelectorUnitTest, ShortRowsSelectScalar) {
    // 10x10 matrix with 1 nnz per row → avg_nnz = 1.0 < 4.0
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 100; i += 10) {
        dense[i] = 1.0f;
    }

    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);

    SpMVConfig config = spmv_auto_config(csr);
    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);

    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, UniformRowsSelectVector) {
    // 10x10 matrix with 5 nnz per row → avg_nnz = 5.0 >= 4.0, skewness = 5/(5+1) < 10
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            dense[i * 10 + j] = 1.0f;
        }
    }

    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);

    SpMVConfig config = spmv_auto_config(csr);
    EXPECT_EQ(config.kernel_type, SpMVConfig::VECTOR_CSR);

    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, SkewedRowsSelectMergePath) {
    // 10x100 matrix: row 0 has 100 nnz, rows 1-9 have 1 nnz each
    // avg_nnz = (100+9)/10 = 10.9 >= 4.0
    // skewness = max/(min+1) = 100/(1+1) = 50.0 >= 10.0
    const int rows = 10;
    const int cols = 100;
    std::vector<float> dense(rows * cols, 0.0f);
    for (int j = 0; j < cols; j++) {
        dense[j] = 1.0f;
    }
    for (int i = 1; i < rows; i++) {
        dense[i * cols] = 1.0f;
    }

    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), rows, cols);

    CSRStats stats = csr_compute_stats(csr);
    ASSERT_GE(stats.avg_nnz_per_row, 4.0f);
    ASSERT_GE(stats.skewness, 10.0f);

    SpMVConfig config = spmv_auto_config(csr);
    EXPECT_EQ(config.kernel_type, SpMVConfig::MERGE_PATH);

    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, NullMatrixFallsBackToSafeDefault) {
    SpMVConfig config = spmv_auto_config(nullptr);

    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    EXPECT_EQ(config.block_size, 256);
}

TEST(KernelSelectorUnitTest, DegenerateMatrixFallsBackToSafeDefault) {
    CSRMatrix* csr = csr_create(3, 4, 0);
    ASSERT_NE(csr, nullptr);

    SpMVConfig config = spmv_auto_config(csr);

    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    EXPECT_EQ(config.block_size, 256);

    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, InvalidMatrixMetadataFallsBackToSafeDefault) {
    CSRMatrix invalid{};
    invalid.num_rows = -1;
    invalid.num_cols = 32;
    invalid.nnz = 4;
    invalid.row_ptrs = nullptr;

    SpMVConfig config = spmv_auto_config(&invalid);

    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    EXPECT_EQ(config.block_size, 256);
}

TEST(KernelSelectorUnitTest, ThresholdsGetSet) {
    SpMVThresholds original = spmv_get_thresholds();

    SpMVThresholds custom(8.0f, 20.0f);
    spmv_set_thresholds(custom);

    SpMVThresholds retrieved = spmv_get_thresholds();
    EXPECT_FLOAT_EQ(retrieved.avg_nnz_threshold, 8.0f);
    EXPECT_FLOAT_EQ(retrieved.skewness_threshold, 20.0f);

    spmv_set_thresholds(original);
}
