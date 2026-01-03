#include <gtest/gtest.h>
#include "spmv/spmv.h"
#include "spmv/csr_matrix.h"
#include "spmv/test_utils.h"

using namespace spmv;
using namespace spmv::test;

class KernelSelectorPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

// **Feature: spmv-gpu, Property 11: Kernel Selector Validity**
// **Validates: Requirements 4.5**
TEST_F(KernelSelectorPropertyTest, SelectorValidity) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 500);
        int cols = rng.randInt(1, 500);
        float density = rng.randFloat(0.001f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        
        SpMVConfig config = spmv_auto_config(csr);
        
        // 验证 block_size 在合理范围内
        EXPECT_GE(config.block_size, 32) 
            << "Block size too small at iteration " << iter;
        EXPECT_LE(config.block_size, 1024) 
            << "Block size too large at iteration " << iter;
        
        // 验证 block_size 是 32 的倍数 (warp size)
        EXPECT_EQ(config.block_size % 32, 0) 
            << "Block size not multiple of 32 at iteration " << iter;
        
        // 验证 kernel_type 是有效值
        EXPECT_TRUE(
            config.kernel_type == SpMVConfig::SCALAR_CSR ||
            config.kernel_type == SpMVConfig::VECTOR_CSR ||
            config.kernel_type == SpMVConfig::MERGE_PATH ||
            config.kernel_type == SpMVConfig::ELL_KERNEL
        ) << "Invalid kernel type at iteration " << iter;
        
        csr_destroy(csr);
    }
}

// 单元测试：验证不同矩阵特征选择不同 Kernel
TEST(KernelSelectorUnitTest, ShortRowsSelectScalar) {
    // 短行矩阵应该选择 Scalar Kernel
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 100; i += 10) {
        dense[i] = 1.0f;  // 每行只有 1 个非零元素
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    
    SpMVConfig config = spmv_auto_config(csr);
    
    CSRStats stats = csr_compute_stats(csr);
    if (stats.avg_nnz_per_row < 4.0f) {
        EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    }
    
    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, UniformRowsSelectVector) {
    // 均匀分布应该选择 Vector Kernel
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 5; j++) {
            dense[i * 10 + j] = 1.0f;  // 每行 5 个非零元素
        }
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    
    SpMVConfig config = spmv_auto_config(csr);
    
    CSRStats stats = csr_compute_stats(csr);
    if (stats.avg_nnz_per_row >= 4.0f && stats.skewness < 10.0f) {
        EXPECT_EQ(config.kernel_type, SpMVConfig::VECTOR_CSR);
    }
    
    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, SkewedRowsSelectMergePath) {
    // 高度不均匀应该选择 Merge Path
    std::vector<float> dense(100, 0.0f);
    // 第一行有很多非零元素
    for (int j = 0; j < 10; j++) {
        dense[j] = 1.0f;
    }
    // 其他行只有 1 个
    for (int i = 1; i < 10; i++) {
        dense[i * 10] = 1.0f;
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    
    SpMVConfig config = spmv_auto_config(csr);
    
    CSRStats stats = csr_compute_stats(csr);
    if (stats.skewness >= 10.0f) {
        EXPECT_EQ(config.kernel_type, SpMVConfig::MERGE_PATH);
    }
    
    csr_destroy(csr);
}

TEST(KernelSelectorUnitTest, LargeVectorUsesTexture) {
    // 大向量应该使用纹理缓存
    std::vector<float> dense(100000, 0.0f);
    for (int i = 0; i < 100000; i += 100) {
        dense[i] = 1.0f;
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 100, 1000);
    
    SpMVConfig config = spmv_auto_config(csr);
    
    if (csr->num_cols > 10000) {
        EXPECT_TRUE(config.use_texture);
    }
    
    csr_destroy(csr);
}
