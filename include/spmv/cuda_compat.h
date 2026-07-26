#ifndef SPMV_CUDA_COMPAT_H
#define SPMV_CUDA_COMPAT_H

#if defined(SPMV_WITH_CUDA) && SPMV_WITH_CUDA

#include <cuda_runtime.h>

#else

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using cudaError_t = int;
using cudaEvent_t = void*;
using cudaStream_t = void*;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidValue = 11;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

struct cudaDeviceProp {
    int memoryClockRate = 0;
    int memoryBusWidth = 0;
};

inline const char* cudaGetErrorString(cudaError_t err) {
    switch (err) {
        case cudaSuccess:
            return "success";
        case cudaErrorMemoryAllocation:
            return "memory allocation failed";
        case cudaErrorInvalidValue:
            return "invalid value";
        default:
            return "cuda unavailable";
    }
}

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (!ptr) return cudaErrorInvalidValue;
    *ptr = (size == 0) ? nullptr : std::malloc(size);
    return (size == 0 || *ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}

inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) {
    if (count > 0 && (!dst || !src)) return cudaErrorInvalidValue;
    if (count > 0) std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* dst, int value, size_t count) {
    if (count > 0 && !dst) return cudaErrorInvalidValue;
    if (count > 0) std::memset(dst, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* dst, int value, size_t count, cudaStream_t = nullptr) {
    return cudaMemset(dst, value, count);
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    if (!prop) return cudaErrorInvalidValue;
    *prop = {};
    return cudaSuccess;
}

inline cudaError_t cudaGetDevice(int* device) {
    if (!device) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}

#endif

#endif  // SPMV_CUDA_COMPAT_H
