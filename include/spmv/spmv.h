#ifndef SPMV_SPMV_H
#define SPMV_SPMV_H

#include "common.h"
#include "csr_matrix.h"
#include "ell_matrix.h"

namespace spmv {

/**
 * @file spmv.h
 * @brief SpMV (Sparse Matrix-Vector Multiplication) operations.
 *
 * Provides CPU and GPU implementations of SpMV with automatic
 * kernel selection based on matrix characteristics.
 */

/// @name CUDA Configuration Constants
/// @{
constexpr int WARP_SIZE = 32;                        ///< CUDA warp size
constexpr int MIN_BLOCK_SIZE = 32;                   ///< Minimum CUDA block size
constexpr int MAX_BLOCK_SIZE = 1024;                 ///< Maximum CUDA block size
constexpr int DEFAULT_BLOCK_SIZE = 256;              ///< Default CUDA block size
constexpr int TEXTURE_CACHE_THRESHOLD_COLS = 10000;  ///< Use texture cache above this
/// @}

/**
 * @brief Thresholds for automatic kernel selection.
 *
 * These can be tuned for different GPU architectures.
 */
struct SpMVThresholds {
    float avg_nnz_threshold;     ///< Below this: use scalar kernel (default: 4.0)
    float skewness_threshold;    ///< Below this: use vector kernel (default: 10.0)
    int texture_cols_threshold;  ///< Above this: use texture cache (default:
                                 ///< 10000)

    SpMVThresholds()
        : avg_nnz_threshold(4.0f),
          skewness_threshold(10.0f),
          texture_cols_threshold(TEXTURE_CACHE_THRESHOLD_COLS) {}

    SpMVThresholds(float avg_nnz, float skewness, int texture_cols)
        : avg_nnz_threshold(avg_nnz),
          skewness_threshold(skewness),
          texture_cols_threshold(texture_cols) {}
};

/**
 * @brief Get current kernel selection thresholds.
 * @return Current threshold values.
 */
SpMVThresholds spmv_get_thresholds();

/**
 * @brief Set kernel selection thresholds.
 * @param thresholds New threshold values.
 */
void spmv_set_thresholds(const SpMVThresholds& thresholds);

/**
 * @brief Configuration for SpMV kernel execution.
 */
struct SpMVConfig {
    /**
     * @brief Kernel type for SpMV operation.
     */
    enum KernelType {
        SCALAR_CSR,  ///< One thread per row (best for very sparse rows)
        VECTOR_CSR,  ///< One warp per row (best for uniform distribution)
        MERGE_PATH,  ///< Load-balanced partitioning (best for skewed matrices)
        ELL_KERNEL   ///< ELL format kernel (column-major coalesced access)
    };

    KernelType kernel_type;  ///< Selected kernel type
    int block_size;          ///< CUDA block size
    bool use_texture;        ///< Use texture cache for x vector

    SpMVConfig() : kernel_type(SCALAR_CSR), block_size(DEFAULT_BLOCK_SIZE), use_texture(false) {}
    SpMVConfig(KernelType kernel_type_, int block_size_, bool use_texture_)
        : kernel_type(kernel_type_), block_size(block_size_), use_texture(use_texture_) {}
};

class SpMVExecutionContext {
   public:
    SpMVExecutionContext() = default;
    ~SpMVExecutionContext() { reset(); }

    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext& operator=(const SpMVExecutionContext&) = delete;

    SpMVExecutionContext(SpMVExecutionContext&& other) noexcept
        : tex_x_(other.tex_x_),
          cached_x_(other.cached_x_),
          cached_x_length_(other.cached_x_length_),
          texture_enabled_(other.texture_enabled_) {
        other.tex_x_ = 0;
        other.cached_x_ = nullptr;
        other.cached_x_length_ = 0;
        other.texture_enabled_ = false;
    }

    SpMVExecutionContext& operator=(SpMVExecutionContext&& other) noexcept {
        if (this != &other) {
            reset();
            tex_x_ = other.tex_x_;
            cached_x_ = other.cached_x_;
            cached_x_length_ = other.cached_x_length_;
            texture_enabled_ = other.texture_enabled_;
            other.tex_x_ = 0;
            other.cached_x_ = nullptr;
            other.cached_x_length_ = 0;
            other.texture_enabled_ = false;
        }
        return *this;
    }

    /** @brief Reset context, freeing texture object. */
    void reset() {
        if (tex_x_ != 0) {
            cudaDestroyTextureObject(tex_x_);
            tex_x_ = 0;
        }
        cached_x_ = nullptr;
        cached_x_length_ = 0;
        texture_enabled_ = false;
    }

    /** @brief Query whether a texture object is currently bound. */
    bool is_texture_bound() const { return tex_x_ != 0; }

   private:
    friend int spmv_prepare_texture(SpMVExecutionContext* context, const float* d_x, size_t x_length,
                                    bool requested, cudaTextureObject_t* tex_out,
                                    bool* use_texture_out);

    cudaTextureObject_t tex_x_ = 0;
    const float* cached_x_ = nullptr;
    size_t cached_x_length_ = 0;
    bool texture_enabled_ = false;
};

/**
 * @brief Result of an SpMV operation.
 */
struct SpMVResult {
    float* y;              ///< Output vector (device pointer)
    float elapsed_ms;      ///< Execution time in milliseconds
    float gflops;          ///< Computed GFLOPS (2 * nnz / time / 1e9)
    float bandwidth_gb_s;  ///< Memory bandwidth in GB/s
    int error_code;        ///< 0 = success, negative = error

    SpMVResult()
        : y(nullptr), elapsed_ms(0.0f), gflops(0.0f), bandwidth_gb_s(0.0f), error_code(0) {}
};

/**
 * @brief CPU reference implementation of CSR SpMV.
 *
 * Computes y = A * x on the CPU. Used for verification.
 *
 * @param A CSR matrix.
 * @param x Input vector (host memory).
 * @param y Output vector (host memory).
 */
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);

/**
 * @brief CPU reference implementation of ELL SpMV.
 *
 * @param A ELL matrix.
 * @param x Input vector (host memory).
 * @param y Output vector (host memory).
 */
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

/**
 * @brief GPU implementation of CSR SpMV.
 *
 * Computes y = A * x on the GPU. All parameters must be in device memory.
 *
 * @param A CSR matrix with device data uploaded.
 * @param d_x Input vector (device memory).
 * @param d_y Output vector (device memory).
 * @param config Kernel configuration (nullptr = auto-select).
 * @param vec_size Size of x vector (-1 = use A->num_cols).
 * @param context Optional execution context for texture caching.
 * @return Result with timing and error code.
 */
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size = -1, SpMVExecutionContext* context = nullptr);

/**
 * @brief GPU implementation of ELL SpMV.
 *
 * @param A ELL matrix with device data uploaded.
 * @param d_x Input vector (device memory).
 * @param d_y Output vector (device memory).
 * @param config Kernel configuration (nullptr = use ELL_KERNEL).
 * @param vec_size Size of x vector (-1 = use A->num_cols).
 * @param context Optional execution context for texture caching.
 * @return Result with timing and error code.
 */
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size = -1, SpMVExecutionContext* context = nullptr);

/**
 * @brief Automatically select optimal kernel configuration.
 *
 * Analyzes matrix structure and chooses the best kernel type
 * based on average nnz per row and row-length skewness.
 *
 * @param A CSR matrix to analyze.
 * @return Recommended configuration.
 */
SpMVConfig spmv_auto_config(const CSRMatrix* A);

/**
 * @brief Validate dimensions for SpMV operation.
 *
 * @param num_cols Matrix column count.
 * @param vec_size Input vector size.
 * @return true if dimensions are compatible.
 */
inline bool spmv_validate_dimensions(int num_cols, int vec_size) {
    return num_cols == vec_size;
}

}  // namespace spmv

#endif  // SPMV_SPMV_H
