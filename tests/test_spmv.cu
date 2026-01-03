#include <gtest/gtest.h>
#include "spmv/spmv.h"
#include "spmv/csr_matrix.h"
#include "spmv/ell_matrix.h"
#include "spmv/cuda_buffer.h"
#include "spmv/test_utils.h"
#include <vector>
#include <cmath>

using namespace spmv;
using namespace spmv::test;

class SpMVPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
    
    bool compareResults(const float* cpu_result, const float* gpu_result, 
                       int size, float rel_tol = 1e-6f) {
        for (int i = 0; i < size; i++) {
            float diff = std::abs(cpu_result[i] - gpu_result[i]);
            float max_val = std::max(std::abs(cpu_result[i]), std::abs(gpu_result[i]));
            
            if (max_val < 1e-10f) {
                // 两个值都接近零
                if (diff > 1e-6f) return false;
            } else {
                float rel_error = diff / max_val;
                if (rel_error > rel_tol) {
                    return false;
                }
            }
        }
        return true;
    }
};

// **Feature: spmv-gpu, Property 8: SpMV CSR Correctness**
// **Validates: Requirements 3.1, 3.3**
TEST_F(SpMVPropertyTest, CSRCorrectness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 200);
        int cols = rng.randInt(1, 200);
        float density = rng.randFloat(0.01f, 0.3f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        auto x = generateRandomVector(cols, rng);
        
        // 创建 CSR 矩阵
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        csr_to_gpu(csr);
        
        // CPU 参考结果
        std::vector<float> y_cpu(rows);
        spmv_cpu_csr(csr, x.data(), y_cpu.data());
        
        // GPU 计算
        CudaBuffer<float> d_x(cols);
        CudaBuffer<float> d_y(rows);
        d_x.copyFromHost(x.data(), cols);
        
        SpMVConfig config;
        config.kernel_type = SpMVConfig::SCALAR_CSR;
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config);
        
        ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS))
            << "SpMV failed at iteration " << iter;
        
        std::vector<float> y_gpu(rows);
        d_y.copyToHost(y_gpu.data(), rows);
        
        EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows))
            << "Results mismatch at iteration " << iter;
        
        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 9: SpMV ELL Correctness**
// **Validates: Requirements 3.2, 3.3**
TEST_F(SpMVPropertyTest, ELLCorrectness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 200);
        int cols = rng.randInt(1, 200);
        float density = rng.randFloat(0.01f, 0.3f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        auto x = generateRandomVector(cols, rng);
        
        // 创建 ELL 矩阵
        ELLMatrix* ell = ell_create(0, 0, 0);
        ell_from_dense(ell, dense.data(), rows, cols);
        ell_to_gpu(ell);
        
        // CPU 参考结果
        std::vector<float> y_cpu(rows);
        spmv_cpu_ell(ell, x.data(), y_cpu.data());
        
        // GPU 计算
        CudaBuffer<float> d_x(cols);
        CudaBuffer<float> d_y(rows);
        d_x.copyFromHost(x.data(), cols);
        
        SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), nullptr);
        
        ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS))
            << "SpMV ELL failed at iteration " << iter;
        
        std::vector<float> y_gpu(rows);
        d_y.copyToHost(y_gpu.data(), rows);
        
        EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows))
            << "ELL results mismatch at iteration " << iter;
        
        ell_destroy(ell);
    }
}

// **Feature: spmv-gpu, Property 10: SpMV Dimension Validation**
// **Validates: Requirements 3.5, 8.5**
TEST_F(SpMVPropertyTest, DimensionValidation) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(5, 50);
        int cols = rng.randInt(5, 50);
        int wrong_cols = cols + rng.randInt(1, 10);
        
        auto dense = generateRandomDenseMatrix(rows, cols, 0.2f, rng);
        auto x_wrong = generateRandomVector(wrong_cols, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        csr_to_gpu(csr);
        
        CudaBuffer<float> d_x(wrong_cols);
        CudaBuffer<float> d_y(rows);
        d_x.copyFromHost(x_wrong.data(), wrong_cols);
        
        // 验证维度不匹配
        EXPECT_FALSE(spmv_validate_dimensions(csr->num_cols, wrong_cols))
            << "Dimension validation should fail";
        
        csr_destroy(csr);
    }
}

// 单元测试
TEST(SpMVUnitTest, EmptyMatrix) {
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_to_gpu(csr);
    
    CudaBuffer<float> d_x(1);
    CudaBuffer<float> d_y(1);
    
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
    
    csr_destroy(csr);
}

TEST(SpMVUnitTest, SingleElement) {
    std::vector<float> dense = {5.0f};
    std::vector<float> x = {2.0f};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 1, 1);
    csr_to_gpu(csr);
    
    std::vector<float> y_cpu(1);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());
    
    CudaBuffer<float> d_x(1);
    CudaBuffer<float> d_y(1);
    d_x.copyFromHost(x.data(), 1);
    
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
    
    std::vector<float> y_gpu(1);
    d_y.copyToHost(y_gpu.data(), 1);
    
    EXPECT_FLOAT_EQ(y_cpu[0], y_gpu[0]);
    EXPECT_FLOAT_EQ(y_gpu[0], 10.0f);
    
    csr_destroy(csr);
}

TEST(SpMVUnitTest, ZeroRows) {
    // 矩阵有全零行
    std::vector<float> dense = {
        1, 2, 0,
        0, 0, 0,  // 全零行
        3, 0, 4
    };
    std::vector<float> x = {1, 1, 1};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);
    
    std::vector<float> y_cpu(3);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());
    
    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), 3);
    
    spmv_csr(csr, d_x.get(), d_y.get(), nullptr);
    
    std::vector<float> y_gpu(3);
    d_y.copyToHost(y_gpu.data(), 3);
    
    EXPECT_FLOAT_EQ(y_gpu[0], 3.0f);
    EXPECT_FLOAT_EQ(y_gpu[1], 0.0f);  // 全零行
    EXPECT_FLOAT_EQ(y_gpu[2], 7.0f);
    
    csr_destroy(csr);
}

TEST(SpMVUnitTest, KernelSelector) {
    // 测试 Kernel 选择器
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 100; i += 10) {
        dense[i] = 1.0f;
    }
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    
    SpMVConfig config = spmv_auto_config(csr);
    
    EXPECT_GE(config.block_size, 32);
    EXPECT_LE(config.block_size, 1024);
    EXPECT_EQ(config.block_size % 32, 0);  // 应该是 32 的倍数
    
    csr_destroy(csr);
}
