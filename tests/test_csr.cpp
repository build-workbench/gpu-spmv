#include <gtest/gtest.h>
#include "spmv/csr_matrix.h"
#include "spmv/test_utils.h"
#include <cstdio>
#include <vector>

using namespace spmv;
using namespace spmv::test;

class CSRPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

// **Feature: spmv-gpu, Property 1: CSR Dense-to-Sparse Round Trip**
// **Validates: Requirements 1.2**
TEST_F(CSRPropertyTest, DenseToSparseRoundTrip) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);
        
        auto dense_original = generateRandomDenseMatrix(rows, cols, density, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        ASSERT_NE(csr, nullptr);
        
        int result = csr_from_dense(csr, dense_original.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        std::vector<float> dense_reconstructed(rows * cols);
        result = csr_to_dense(csr, dense_reconstructed.data());
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        EXPECT_TRUE(floatArraysEqual(dense_original.data(), 
                                     dense_reconstructed.data(), 
                                     rows * cols))
            << "Round trip failed at iteration " << iter;
        
        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 2: CSR Element Lookup Correctness**
// **Validates: Requirements 1.3**
TEST_F(CSRPropertyTest, ElementLookupCorrectness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 50);
        int cols = rng.randInt(1, 50);
        float density = rng.randFloat(0.1f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        ASSERT_NE(csr, nullptr);
        
        int result = csr_from_dense(csr, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 随机查询多个位置
        for (int q = 0; q < 20; q++) {
            int r = rng.randInt(0, rows - 1);
            int c = rng.randInt(0, cols - 1);
            
            float expected = dense[r * cols + c];
            float actual = csr_get_element(csr, r, c);
            
            EXPECT_FLOAT_EQ(expected, actual)
                << "Element lookup failed at (" << r << ", " << c << ") "
                << "iteration " << iter;
        }
        
        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 3: CSR Serialization Round Trip**
// **Validates: Requirements 1.5**
TEST_F(CSRPropertyTest, SerializationRoundTrip) {
    const char* test_file = "/tmp/csr_test.bin";
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        CSRMatrix* csr_original = csr_create(0, 0, 0);
        ASSERT_NE(csr_original, nullptr);
        
        int result = csr_from_dense(csr_original, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 序列化
        result = csr_serialize(csr_original, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 反序列化
        CSRMatrix* csr_loaded = csr_create(0, 0, 0);
        result = csr_deserialize(csr_loaded, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 验证
        EXPECT_EQ(csr_original->num_rows, csr_loaded->num_rows);
        EXPECT_EQ(csr_original->num_cols, csr_loaded->num_cols);
        EXPECT_EQ(csr_original->nnz, csr_loaded->nnz);
        
        if (csr_original->nnz > 0) {
            EXPECT_TRUE(floatArraysEqual(csr_original->values, 
                                         csr_loaded->values, 
                                         csr_original->nnz));
            EXPECT_TRUE(intArraysEqual(csr_original->col_indices,
                                       csr_loaded->col_indices,
                                       csr_original->nnz));
        }
        EXPECT_TRUE(intArraysEqual(csr_original->row_ptrs,
                                   csr_loaded->row_ptrs,
                                   csr_original->num_rows + 1));
        
        csr_destroy(csr_original);
        csr_destroy(csr_loaded);
    }
    
    std::remove(test_file);
}

// 单元测试：边界情况
TEST(CSRUnitTest, EmptyMatrix) {
    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    EXPECT_EQ(csr->num_rows, 0);
    EXPECT_EQ(csr->num_cols, 0);
    EXPECT_EQ(csr->nnz, 0);
    csr_destroy(csr);
}

TEST(CSRUnitTest, AllZeroMatrix) {
    std::vector<float> dense(9, 0.0f);  // 3x3 全零矩阵
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 3, 3);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    EXPECT_EQ(csr->num_rows, 3);
    EXPECT_EQ(csr->num_cols, 3);
    EXPECT_EQ(csr->nnz, 0);
    
    csr_destroy(csr);
}

TEST(CSRUnitTest, SingleElementMatrix) {
    std::vector<float> dense = {5.0f};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 1, 1);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    EXPECT_EQ(csr->num_rows, 1);
    EXPECT_EQ(csr->num_cols, 1);
    EXPECT_EQ(csr->nnz, 1);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 5.0f);
    
    csr_destroy(csr);
}

TEST(CSRUnitTest, GPUTransfer) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};  // 3x3
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 3, 3);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    // 传输到 GPU
    result = csr_to_gpu(csr);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    EXPECT_NE(csr->d_values, nullptr);
    EXPECT_NE(csr->d_col_indices, nullptr);
    EXPECT_NE(csr->d_row_ptrs, nullptr);
    
    // 修改主机数据
    for (int i = 0; i < csr->nnz; i++) {
        csr->values[i] = 0.0f;
    }
    
    // 从 GPU 传回
    result = csr_from_gpu(csr);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    // 验证数据恢复
    std::vector<float> reconstructed(9);
    csr_to_dense(csr, reconstructed.data());
    
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(dense[i], reconstructed[i]);
    }
    
    csr_destroy(csr);
}
