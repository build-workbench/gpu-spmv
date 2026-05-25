#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"

#include <gtest/gtest.h>
#include <vector>

using namespace spmv;

#if !SPMV_WITH_CUDA

TEST(NoCudaModeTest, SpMVCsrFailsGracefullyWithoutCudaBackend) {
    std::vector<float> dense = {1.0f, 0.0f, 0.0f, 2.0f};
    std::vector<float> x = {3.0f, 4.0f};
    std::vector<float> y(2, 0.0f);

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 2, 2), static_cast<int>(SpMVError::SUCCESS));

    SpMVResult result = spmv_csr(csr, x.data(), y.data(), nullptr, 2);

    EXPECT_EQ(result.error_code, static_cast<int>(SpMVError::KERNEL_LAUNCH));
    EXPECT_EQ(result.y, y.data());

    csr_destroy(csr);
}

#endif
