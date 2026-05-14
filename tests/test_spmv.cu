#include "spmv/csr_matrix.h"
#include "spmv/cuda_buffer.h"
#include "spmv/ell_matrix.h"
#include "spmv/spmv.h"
#include "spmv/test_utils.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace spmv;
using namespace spmv::test;

static bool compareResults(const float* cpu_result, const float* gpu_result, int size,
                           float rel_tol = 1e-6f) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(cpu_result[i] - gpu_result[i]);
        float max_val = std::max(std::abs(cpu_result[i]), std::abs(gpu_result[i]));

        if (max_val < 1e-10f) {
            if (diff > 1e-6f)
                return false;
        } else {
            float rel_error = diff / max_val;
            if (rel_error > rel_tol) {
                return false;
            }
        }
    }
    return true;
}

class SpMVPropertyTest : public ::testing::Test {
   protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
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

        std::vector<SpMVConfig> configs = {SpMVConfig(SpMVConfig::SCALAR_CSR, 256, false),
                                           SpMVConfig(SpMVConfig::VECTOR_CSR, 256, false),
                                           SpMVConfig(SpMVConfig::MERGE_PATH, 256, false)};

        for (const auto& config : configs) {
            SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
            ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS))
                << "SpMV failed at iteration " << iter << " kernel " << config.kernel_type;

            std::vector<float> y_gpu(rows);
            d_y.copyToHost(y_gpu.data(), rows);

            EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows))
                << "Results mismatch at iteration " << iter << " kernel " << config.kernel_type;
        }

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

        SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), nullptr, cols);

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
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(0);
    CudaBuffer<float> d_y(0);

    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr, 0);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
    EXPECT_EQ(result.y, d_y.get());

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

    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr, static_cast<int>(x.size()));
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(1);
    d_y.copyToHost(y_gpu.data(), 1);

    EXPECT_FLOAT_EQ(y_cpu[0], y_gpu[0]);
    EXPECT_FLOAT_EQ(y_gpu[0], 10.0f);

    csr_destroy(csr);
}

TEST(SpMVUnitTest, ZeroRows) {
    // 矩阵有全零行
    std::vector<float> dense = {1, 2, 0, 0, 0, 0,  // 全零行
                                3, 0, 4};
    std::vector<float> x = {1, 1, 1};

    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    std::vector<float> y_cpu(3);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());

    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), 3);

    spmv_csr(csr, d_x.get(), d_y.get(), nullptr, static_cast<int>(x.size()));

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

TEST(SpMVUnitTest, ExecutionContextReusesTexture) {
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 10; ++i) {
        dense[i * 10 + i] = 1.0f;
    }
    std::vector<float> x(10, 2.0f);

    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    csr_to_gpu(csr);

    CudaBuffer<float> d_x(10);
    CudaBuffer<float> d_y(10);
    d_x.copyFromHost(x.data(), 10);

    SpMVConfig config(SpMVConfig::SCALAR_CSR, 256, true);
    SpMVExecutionContext context;

    SpMVResult first = spmv_csr(csr, d_x.get(), d_y.get(), &config, 10, &context);
    ASSERT_EQ(first.error_code, static_cast<int>(SpMVError::SUCCESS));
    EXPECT_TRUE(context.is_texture_bound());

    SpMVResult second = spmv_csr(csr, d_x.get(), d_y.get(), &config, 10, &context);
    ASSERT_EQ(second.error_code, static_cast<int>(SpMVError::SUCCESS));
    EXPECT_TRUE(context.is_texture_bound());

    std::vector<float> y_gpu(10);
    d_y.copyToHost(y_gpu.data(), 10);
    for (float value : y_gpu) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    csr_destroy(csr);
}

TEST(SpMVUnitTest, InvalidBlockSizeRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), 3);

    SpMVConfig config(SpMVConfig::VECTOR_CSR, 48, false);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, InvalidELLKernelRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    ELLMatrix* ell = ell_create(0, 0, 0);
    ASSERT_EQ(ell_from_dense(ell, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(ell_to_gpu(ell), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), 3);

    SpMVConfig config(SpMVConfig::VECTOR_CSR, 256, false);
    SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), &config, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    ell_destroy(ell);
}

TEST(SpMVUnitTest, ZeroNnzMatricesProduceZeroOutputForAllCSRKernels) {
    CSRMatrix* csr = csr_create(4, 4, 0);
    ASSERT_NE(csr, nullptr);
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> x = {1.0f, -2.0f, 3.0f, 4.0f};
    CudaBuffer<float> d_x(4);
    CudaBuffer<float> d_y(4);
    d_x.copyFromHost(x.data(), x.size());

    std::vector<SpMVConfig> configs = {SpMVConfig(SpMVConfig::SCALAR_CSR, 256, false),
                                       SpMVConfig(SpMVConfig::VECTOR_CSR, 256, false),
                                       SpMVConfig(SpMVConfig::MERGE_PATH, 256, false)};

    for (const auto& config : configs) {
        ASSERT_EQ(cudaMemset(d_y.get(), 0x7f, 4 * sizeof(float)), cudaSuccess);
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 4);
        ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

        std::vector<float> y_gpu(4, 1.0f);
        d_y.copyToHost(y_gpu.data(), y_gpu.size());
        for (float value : y_gpu) {
            EXPECT_FLOAT_EQ(value, 0.0f);
        }
    }

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MergePathHandlesHighlySkewedRows) {
    const int rows = 8;
    const int cols = 16;
    std::vector<float> dense(rows * cols, 0.0f);

    for (int j = 0; j < cols; ++j) {
        dense[j] = static_cast<float>(j + 1);
    }
    dense[1 * cols + 0] = 2.0f;
    dense[3 * cols + 3] = -1.5f;
    dense[5 * cols + 5] = 4.0f;
    dense[7 * cols + 2] = 0.5f;

    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = static_cast<float>((i % 5) - 2);
    }

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), rows, cols), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_cpu(rows, 0.0f);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());

    CudaBuffer<float> d_x(cols);
    CudaBuffer<float> d_y(rows);
    d_x.copyFromHost(x.data(), x.size());

    SpMVConfig config(SpMVConfig::MERGE_PATH, 256, false);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(rows, 0.0f);
    d_y.copyToHost(y_gpu.data(), y_gpu.size());
    EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MergePathHandlesInterleavedEmptyRows) {
    const int rows = 6;
    const int cols = 6;
    std::vector<float> dense = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 6, 0, 0};
    std::vector<float> x = {1, 2, 3, 4, 5, 6};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), rows, cols), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_cpu(rows, 0.0f);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());

    CudaBuffer<float> d_x(cols);
    CudaBuffer<float> d_y(rows);
    d_x.copyFromHost(x.data(), x.size());

    SpMVConfig config(SpMVConfig::MERGE_PATH, 256, false);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, cols);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(rows, 0.0f);
    d_y.copyToHost(y_gpu.data(), y_gpu.size());
    EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MissingInputVectorRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_y(3);
    SpMVResult result = spmv_csr(csr, nullptr, d_y.get(), nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MissingOutputVectorRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_csr(csr, d_x.get(), nullptr, nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, ZeroLengthVectorProducesZeroOutput) {
    CSRMatrix* csr = csr_create(3, 0, 0);
    ASSERT_NE(csr, nullptr);
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_y(3);
    ASSERT_EQ(cudaMemset(d_y.get(), 0x55, 3 * sizeof(float)), cudaSuccess);

    SpMVConfig config(SpMVConfig::SCALAR_CSR, 256, false);
    SpMVResult result = spmv_csr(csr, nullptr, d_y.get(), &config, 0);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(3, 1.0f);
    d_y.copyToHost(y_gpu.data(), y_gpu.size());
    for (float value : y_gpu) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    csr_destroy(csr);
}

TEST(SpMVUnitTest, AutoConfigHandlesNullInput) {
    SpMVConfig config = spmv_auto_config(nullptr);

    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    EXPECT_EQ(config.block_size, 256);
    EXPECT_FALSE(config.use_texture);
}

TEST(SpMVUnitTest, BenchmarkAutoConfigReturnsSafeDefaultForDegenerateMatrix) {
    CSRMatrix* csr = csr_create(2, 2, 0);
    ASSERT_NE(csr, nullptr);

    SpMVConfig config = spmv_auto_config(csr);

    EXPECT_EQ(config.kernel_type, SpMVConfig::SCALAR_CSR);
    EXPECT_EQ(config.block_size, 256);
    EXPECT_FALSE(config.use_texture);

    csr_destroy(csr);
}

TEST(SpMVUnitTest, InvalidVectorDimensionRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1, 1};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(4);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr, 4);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_DIMENSION));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MissingELLInputVectorRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};

    ELLMatrix* ell = ell_create(0, 0, 0);
    ASSERT_EQ(ell_from_dense(ell, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(ell_to_gpu(ell), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_y(3);
    SpMVResult result = spmv_ell(ell, nullptr, d_y.get(), nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    ell_destroy(ell);
}

TEST(SpMVUnitTest, MissingELLOutputVectorRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    ELLMatrix* ell = ell_create(0, 0, 0);
    ASSERT_EQ(ell_from_dense(ell, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(ell_to_gpu(ell), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_ell(ell, d_x.get(), nullptr, nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_ARGUMENT));

    ell_destroy(ell);
}

TEST(SpMVUnitTest, ZeroLengthVectorProducesZeroOutputForELL) {
    ELLMatrix* ell = ell_create(3, 0, 0);
    ASSERT_NE(ell, nullptr);
    ASSERT_EQ(ell_to_gpu(ell), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_y(3);
    ASSERT_EQ(cudaMemset(d_y.get(), 0x55, 3 * sizeof(float)), cudaSuccess);

    SpMVResult result = spmv_ell(ell, nullptr, d_y.get(), nullptr, 0);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(3, 1.0f);
    d_y.copyToHost(y_gpu.data(), y_gpu.size());
    for (float value : y_gpu) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    ell_destroy(ell);
}

TEST(SpMVUnitTest, InvalidELLVectorDimensionRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1, 1};

    ELLMatrix* ell = ell_create(0, 0, 0);
    ASSERT_EQ(ell_from_dense(ell, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(ell_to_gpu(ell), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(4);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), nullptr, 4);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_DIMENSION));

    ell_destroy(ell);
}

TEST(SpMVUnitTest, MissingUploadedCSRRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_FORMAT));

    csr_destroy(csr);
}

TEST(SpMVUnitTest, MissingUploadedELLRejected) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};

    ELLMatrix* ell = ell_create(0, 0, 0);
    ASSERT_EQ(ell_from_dense(ell, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));

    CudaBuffer<float> d_x(3);
    CudaBuffer<float> d_y(3);
    d_x.copyFromHost(x.data(), x.size());

    SpMVResult result = spmv_ell(ell, d_x.get(), d_y.get(), nullptr, 3);
    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::INVALID_FORMAT));

    ell_destroy(ell);
}

TEST(SpMVUnitTest, MergePathTexturePathMatchesCpuReference) {
    const int rows = 5;
    const int cols = 11001;
    std::vector<float> dense(rows * cols, 0.0f);
    for (int j = 0; j < cols; ++j) {
        dense[j] = 1.0f;
    }
    dense[1 * cols + 1] = 2.0f;
    dense[3 * cols + 10999] = -3.0f;

    std::vector<float> x(cols, 1.0f);

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), rows, cols), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_cpu(rows, 0.0f);
    spmv_cpu_csr(csr, x.data(), y_cpu.data());

    CudaBuffer<float> d_x(cols);
    CudaBuffer<float> d_y(rows);
    d_x.copyFromHost(x.data(), x.size());

    SpMVConfig config(SpMVConfig::MERGE_PATH, 256, true);
    SpMVExecutionContext context;
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, cols, &context);
    ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));

    std::vector<float> y_gpu(rows, 0.0f);
    d_y.copyToHost(y_gpu.data(), y_gpu.size());
    EXPECT_TRUE(compareResults(y_cpu.data(), y_gpu.data(), rows));

    csr_destroy(csr);
}
