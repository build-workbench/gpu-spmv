#ifndef SPMV_SPMV_H
#define SPMV_SPMV_H

#include "common.h"
#include "csr_matrix.h"
#include "ell_matrix.h"

namespace spmv {

/// @name CUDA Configuration Constants
/// @{
constexpr int WARP_SIZE = 32;
constexpr int MIN_BLOCK_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int DEFAULT_BLOCK_SIZE = 256;
/// @}

struct SpMVThresholds {
    float avg_nnz_threshold;
    float skewness_threshold;

    SpMVThresholds() : avg_nnz_threshold(4.0f), skewness_threshold(10.0f) {}
    SpMVThresholds(float avg_nnz, float skewness)
        : avg_nnz_threshold(avg_nnz), skewness_threshold(skewness) {}
};

SpMVThresholds spmv_get_thresholds();
void spmv_set_thresholds(const SpMVThresholds& thresholds);

struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,
        VECTOR_CSR,
        MERGE_PATH,
        ELL_KERNEL
    };

    KernelType kernel_type;
    int block_size;

    SpMVConfig() : kernel_type(SCALAR_CSR), block_size(DEFAULT_BLOCK_SIZE) {}
    SpMVConfig(KernelType kernel_type_, int block_size_)
        : kernel_type(kernel_type_), block_size(block_size_) {}
};

struct SpMVResult {
    float* y;
    float elapsed_ms;
    float gflops;
    float bandwidth_gb_s;
    int error_code;

    SpMVResult()
        : y(nullptr), elapsed_ms(0.0f), gflops(0.0f), bandwidth_gb_s(0.0f), error_code(0) {}
};

void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);

SpMVConfig spmv_auto_config(const CSRMatrix* A);

inline bool spmv_validate_dimensions(int num_cols, int vec_size) {
    return num_cols == vec_size;
}

}  // namespace spmv

#endif  // SPMV_SPMV_H
