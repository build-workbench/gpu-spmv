#include <gtest/gtest.h>
#include "spmv/ell_matrix.h"
#include "spmv/csr_matrix.h"
#include "spmv/test_utils.h"
#include <cstdio>
#include <vector>

using namespace spmv;
using namespace spmv::test;

class ELLPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

// **Feature: spmv-gpu, Property 4: ELL Dense-to-Sparse Round Trip**
// **Validates: Requirements 2.2**
TEST_F(ELLPropertyTest, DenseToSparseRoundTrip) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);
        
        auto dense_original = generateRandomDenseMatrix(rows, cols, density, rng);
        
        ELLMatrix* ell = ell_create(0, 0, 0);
        ASSERT_NE(ell, nullptr);
        
        int result = ell_from_dense(ell, dense_original.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        std::vector<float> dense_reconstructed(rows * cols);
        result = ell_to_dense(ell, dense_reconstructed.data());
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        EXPECT_TRUE(floatArraysEqual(dense_original.data(), 
                                     dense_reconstructed.data(), 
                                     rows * cols))
            << "Round trip failed at iteration " << iter;
        
        ell_destroy(ell);
    }
}

// **Feature: spmv-gpu, Property 5: ELL Padding Correctness**
// **Validates: Requirements 2.3**
TEST_F(ELLPropertyTest, PaddingCorrectness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(2, 50);
        int cols = rng.randInt(2, 50);
        float density = rng.randFloat(0.1f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        ELLMatrix* ell = ell_create(0, 0, 0);
        int result = ell_from_dense(ell, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 验证每行的填充
        for (int i = 0; i < rows; i++) {
            // 计算该行实际非零元素数
            int actual_nnz = 0;
            for (int j = 0; j < cols; j++) {
                if (dense[i * cols + j] != 0.0f) actual_nnz++;
            }
            
            // 验证填充位置
            for (int k = actual_nnz; k < ell->max_nnz_per_row; k++) {
                int idx = ell_index(i, k, rows);
                EXPECT_EQ(ell->col_indices[idx], -1)
                    << "Padding col_index should be -1 at row " << i << " k " << k;
                EXPECT_FLOAT_EQ(ell->values[idx], 0.0f)
                    << "Padding value should be 0 at row " << i << " k " << k;
            }
        }
        
        ell_destroy(ell);
    }
}

// **Feature: spmv-gpu, Property 6: ELL Column-Major Layout**
// **Validates: Requirements 2.4**
TEST_F(ELLPropertyTest, ColumnMajorLayout) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(2, 50);
        int cols = rng.randInt(2, 50);
        float density = rng.randFloat(0.1f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        ELLMatrix* ell = ell_create(0, 0, 0);
        int result = ell_from_dense(ell, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        // 验证 Column-major 布局: element(row, k) 在 index k * num_rows + row
        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < ell->max_nnz_per_row; k++) {
                int expected_idx = k * rows + i;
                int actual_idx = ell_index(i, k, rows);
                EXPECT_EQ(expected_idx, actual_idx)
                    << "Column-major index mismatch at row " << i << " k " << k;
            }
        }
        
        ell_destroy(ell);
    }
}

// **Feature: spmv-gpu, Property 7: ELL Serialization Round Trip**
// **Validates: Requirements 2.5**
TEST_F(ELLPropertyTest, SerializationRoundTrip) {
    const char* test_file = "/tmp/ell_test.bin";
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        
        ELLMatrix* ell_original = ell_create(0, 0, 0);
        int result = ell_from_dense(ell_original, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        result = ell_serialize(ell_original, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        ELLMatrix* ell_loaded = ell_create(0, 0, 0);
        result = ell_deserialize(ell_loaded, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
        
        EXPECT_EQ(ell_original->num_rows, ell_loaded->num_rows);
        EXPECT_EQ(ell_original->num_cols, ell_loaded->num_cols);
        EXPECT_EQ(ell_original->max_nnz_per_row, ell_loaded->max_nnz_per_row);
        
        size_t size = static_cast<size_t>(ell_original->num_rows) * ell_original->max_nnz_per_row;
        if (size > 0) {
            EXPECT_TRUE(floatArraysEqual(ell_original->values, 
                                         ell_loaded->values, size));
            EXPECT_TRUE(intArraysEqual(ell_original->col_indices,
                                       ell_loaded->col_indices, size));
        }
        
        ell_destroy(ell_original);
        ell_destroy(ell_loaded);
    }
    
    std::remove(test_file);
}

// 单元测试
TEST(ELLUnitTest, FromCSR) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};  // 3x3
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    
    ELLMatrix* ell = ell_create(0, 0, 0);
    int result = ell_from_csr(ell, csr);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    std::vector<float> reconstructed(9);
    ell_to_dense(ell, reconstructed.data());
    
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(dense[i], reconstructed[i]);
    }
    
    csr_destroy(csr);
    ell_destroy(ell);
}

TEST(ELLUnitTest, GPUTransfer) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};  // 3x3
    
    ELLMatrix* ell = ell_create(0, 0, 0);
    ell_from_dense(ell, dense.data(), 3, 3);
    
    int result = ell_to_gpu(ell);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    // 清除主机数据
    size_t size = static_cast<size_t>(ell->num_rows) * ell->max_nnz_per_row;
    for (size_t i = 0; i < size; i++) {
        ell->values[i] = 0.0f;
    }
    
    result = ell_from_gpu(ell);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    
    std::vector<float> reconstructed(9);
    ell_to_dense(ell, reconstructed.data());
    
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(dense[i], reconstructed[i]);
    }
    
    ell_destroy(ell);
}
