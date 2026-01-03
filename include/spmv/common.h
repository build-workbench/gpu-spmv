#ifndef SPMV_COMMON_H
#define SPMV_COMMON_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace spmv {

// 错误码定义
enum class SpMVError {
    SUCCESS = 0,
    INVALID_DIMENSION = -1,
    CUDA_MALLOC = -2,
    CUDA_MEMCPY = -3,
    KERNEL_LAUNCH = -4,
    INVALID_FORMAT = -5,
    FILE_IO = -6,
    OUT_OF_MEMORY = -7,
    INVALID_ARGUMENT = -8
};

// 错误码转字符串
inline const char* spmv_error_string(SpMVError err) {
    switch (err) {
        case SpMVError::SUCCESS: return "Success";
        case SpMVError::INVALID_DIMENSION: return "Invalid matrix/vector dimension";
        case SpMVError::CUDA_MALLOC: return "CUDA memory allocation failed";
        case SpMVError::CUDA_MEMCPY: return "CUDA memory copy failed";
        case SpMVError::KERNEL_LAUNCH: return "CUDA kernel launch failed";
        case SpMVError::INVALID_FORMAT: return "Invalid sparse matrix format";
        case SpMVError::FILE_IO: return "File I/O error";
        case SpMVError::OUT_OF_MEMORY: return "Out of memory";
        case SpMVError::INVALID_ARGUMENT: return "Invalid argument";
        default: return "Unknown error";
    }
}

// CUDA 异常类
class CudaException : public std::runtime_error {
public:
    explicit CudaException(cudaError_t err)
        : std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)),
          error_(err) {}
    cudaError_t error() const { return error_; }
private:
    cudaError_t error_;
};

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return static_cast<int>(spmv::SpMVError::CUDA_MALLOC); \
    } \
} while(0)

#define CUDA_CHECK_THROW(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw spmv::CudaException(err); \
    } \
} while(0)

} // namespace spmv

#endif // SPMV_COMMON_H
