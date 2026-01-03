#include <gtest/gtest.h>
#include "spmv/bandwidth.h"
#include "spmv/spmv.h"
#include "spmv/csr_matrix.h"
#include "spmv/cuda_buffer.h"
#include "spmv/test_utils.h"

using namespace spmv;
using namespace spmv::test;

class BandwidthPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

// **Feature: spmv-gpu, Property 12: Bandwidth Metrics Validity**
// **Validates: Requirements 5.5**
TEST_F(BandwidthPropertyTest, MetricsValidity) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(10, 200);
        int cols = rng.randInt(10, 200);
        float density = rng.randFloat(0.05f, 0.3f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        auto x = generateRandomVector(cols, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        csr_to_gpu(csr);
        
        CudaBuffer<float> d_x(cols);
        CudaBuffer<float> d_y(rows);
        d_x.copyFromHost(x.data(), cols);
        
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), nullptr);
        
        ASSERT_EQ(result.error_code, static_cast<int>(SpMVError::SUCCESS));
        
        // 验证带宽度量有效性
        EXPECT_GE(result.bandwidth_gb_s, 0.0f)
            << "Achieved bandwidth should be non-negative at iteration " << iter;
        
        BandwidthMetrics bw = compute_bandwidth_csr(csr, result.elapsed_ms);
        
        EXPECT_GT(bw.theoretical_bandwidth_gb_s, 0.0f)
            << "Theoretical bandwidth should be positive at iteration " << iter;
        
        EXPECT_GE(bw.efficiency, 0.0f)
            << "Efficiency should be non-negative at iteration " << iter;
        
        EXPECT_LE(bw.efficiency, 1.0f)
            << "Efficiency should not exceed 1.0 at iteration " << iter;
        
        csr_destroy(csr);
    }
}

// 单元测试
TEST(BandwidthUnitTest, PeakBandwidth) {
    float peak_bw = get_gpu_peak_bandwidth();
    EXPECT_GT(peak_bw, 0.0f);
    EXPECT_LT(peak_bw, 10000.0f);  // 合理的上限 (10 TB/s)
}

TEST(BandwidthUnitTest, CSRBandwidthCalculation) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    
    float elapsed_ms = 1.0f;  // 假设 1ms
    BandwidthMetrics bw = compute_bandwidth_csr(csr, elapsed_ms);
    
    EXPECT_GT(bw.achieved_bandwidth_gb_s, 0.0f);
    EXPECT_GT(bw.theoretical_bandwidth_gb_s, 0.0f);
    EXPECT_GE(bw.efficiency, 0.0f);
    EXPECT_LE(bw.efficiency, 1.0f);
    
    csr_destroy(csr);
}

TEST(BandwidthUnitTest, ELLBandwidthCalculation) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    
    ELLMatrix* ell = ell_create(0, 0, 0);
    ell_from_dense(ell, dense.data(), 3, 3);
    
    float elapsed_ms = 1.0f;
    BandwidthMetrics bw = compute_bandwidth_ell(ell, elapsed_ms);
    
    EXPECT_GT(bw.achieved_bandwidth_gb_s, 0.0f);
    EXPECT_GT(bw.theoretical_bandwidth_gb_s, 0.0f);
    EXPECT_GE(bw.efficiency, 0.0f);
    EXPECT_LE(bw.efficiency, 1.0f);
    
    ell_destroy(ell);
}

TEST(BandwidthUnitTest, ZeroElapsedTime) {
    std::vector<float> dense = {1, 0, 2};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 1, 3);
    
    BandwidthMetrics bw = compute_bandwidth_csr(csr, 0.0f);
    
    // 零时间应该返回零度量
    EXPECT_EQ(bw.achieved_bandwidth_gb_s, 0.0f);
    EXPECT_EQ(bw.efficiency, 0.0f);
    
    csr_destroy(csr);
}
