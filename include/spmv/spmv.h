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
    enum KernelType { SCALAR_CSR, VECTOR_CSR, MERGE_PATH, ELL_KERNEL };

    KernelType kernel_type;
    int block_size;
    /// When true (default), spmv_csr/spmv_ell block until d_y is complete and
    /// populate elapsed_ms/gflops/bandwidth_gb_s.  When false, the kernel is
    /// only enqueued on the stream — no CUDA events, no synchronization — so
    /// calls can be pipelined asynchronously.  In that mode the timing fields
    /// stay zero and only synchronous launch failures are reported; the caller
    /// must synchronize the stream before reading d_y.
    bool enable_timing;

    SpMVConfig() : kernel_type(SCALAR_CSR), block_size(DEFAULT_BLOCK_SIZE), enable_timing(true) {}
    SpMVConfig(KernelType kernel_type_, int block_size_, bool enable_timing_ = true)
        : kernel_type(kernel_type_), block_size(block_size_), enable_timing(enable_timing_) {}
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

/// Typed accessor for SpMVResult::error_code.
inline SpMVError spmv_result_error(const SpMVResult& result) {
    return static_cast<SpMVError>(result.error_code);
}

/// CPU reference implementations.  Return 0 on success, negative error code
/// on invalid arguments or malformed matrix storage.
int spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
int spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

/**
 * @brief Compute y = A * x on the GPU (CSR).
 *
 * Synchronization: with config->enable_timing (the default) the call blocks
 * until d_y is complete and reports timing metrics.  With enable_timing set
 * to false the kernel is only enqueued on `stream` and the call returns
 * immediately; synchronize the stream yourself before reading d_y.
 */
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);

/// Compute y = A * x on the GPU (ELL).  Synchronization semantics match
/// spmv_csr; see above.
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);

/// Pick a CSR kernel and block size from matrix statistics.
SpMVConfig spmv_auto_config(const CSRMatrix* A);

/// Pick a configuration for ELL matrices (single kernel; provided for API
/// symmetry with spmv_auto_config).
SpMVConfig spmv_auto_config_ell(const ELLMatrix* A);

inline bool spmv_validate_dimensions(int num_cols, int vec_size) {
    return num_cols == vec_size;
}

}  // namespace spmv

#endif  // SPMV_SPMV_H
