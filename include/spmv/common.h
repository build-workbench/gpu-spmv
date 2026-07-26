#ifndef SPMV_COMMON_H
#define SPMV_COMMON_H

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cuda_compat.h"

namespace spmv {

/**
 * @file common.h
 * @brief Common definitions, error codes, and CUDA utilities.
 */

/**
 * @brief Error codes for SpMV operations.
 *
 * These error codes are returned by most SpMV functions to indicate
 * success or failure of the operation.
 */
enum class SpMVError {
    SUCCESS = 0,             ///< Operation completed successfully
    INVALID_DIMENSION = -1,  ///< Matrix or vector dimensions are invalid
    CUDA_MALLOC = -2,        ///< CUDA memory allocation failed
    CUDA_MEMCPY = -3,        ///< CUDA memory copy failed
    KERNEL_LAUNCH = -4,      ///< CUDA kernel launch failed
    INVALID_FORMAT = -5,     ///< Sparse matrix format is invalid
    FILE_IO = -6,            ///< File I/O error
    OUT_OF_MEMORY = -7,      ///< Host memory allocation failed
    INVALID_ARGUMENT = -8    ///< Invalid argument passed to function
};

/**
 * @brief Convert error code to human-readable string.
 *
 * @param err The error code to convert.
 * @return A string describing the error.
 */
inline const char* spmv_error_string(SpMVError err) {
    switch (err) {
        case SpMVError::SUCCESS:
            return "Success";
        case SpMVError::INVALID_DIMENSION:
            return "Invalid matrix/vector dimension";
        case SpMVError::CUDA_MALLOC:
            return "CUDA memory allocation failed";
        case SpMVError::CUDA_MEMCPY:
            return "CUDA memory copy failed";
        case SpMVError::KERNEL_LAUNCH:
            return "CUDA kernel launch failed";
        case SpMVError::INVALID_FORMAT:
            return "Invalid sparse matrix format";
        case SpMVError::FILE_IO:
            return "File I/O error";
        case SpMVError::OUT_OF_MEMORY:
            return "Out of memory";
        case SpMVError::INVALID_ARGUMENT:
            return "Invalid argument";
        default:
            return "Unknown error";
    }
}

/**
 * @brief Exception class for CUDA errors.
 *
 * Thrown when a CUDA operation fails. Contains the CUDA error code
 * and a descriptive message.
 */
class CudaException : public std::runtime_error {
   public:
    /**
     * @brief Construct a CudaException from a CUDA error code.
     * @param err The CUDA error code.
     */
    explicit CudaException(cudaError_t err)
        : std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)), error_(err) {}

    /**
     * @brief Get the CUDA error code.
     * @return The CUDA error code.
     */
    cudaError_t error() const { return error_; }

   private:
    cudaError_t error_;
};

#define SPMV_CUDA_CHECK(call, error_code)                                              \
    do {                                                                               \
        cudaError_t spmv_err_ = (call);                                                \
        if (spmv_err_ != cudaSuccess) {                                                \
            fprintf(stderr, "[gpu-spmv] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(spmv_err_));                                    \
            return static_cast<int>(error_code);                                       \
        }                                                                              \
    } while (0)

#define SPMV_CUDA_CHECK_THROW(call)           \
    do {                                      \
        cudaError_t spmv_err_ = (call);       \
        if (spmv_err_ != cudaSuccess) {       \
            throw spmv::CudaException(spmv_err_); \
        }                                     \
    } while (0)

}  // namespace spmv

#endif  // SPMV_COMMON_H
