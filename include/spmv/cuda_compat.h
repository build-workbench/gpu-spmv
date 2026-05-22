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
using cudaTextureObject_t = std::uintptr_t;
using cudaEvent_t = void*;

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorMemoryAllocation = 2;
constexpr cudaError_t cudaErrorInvalidValue = 11;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

enum {
    cudaResourceTypeLinear = 0,
    cudaAddressModeClamp = 0,
    cudaFilterModePoint = 0,
    cudaReadModeElementType = 0
};

struct cudaChannelFormatDesc {
    int x = 0;
    int y = 0;
    int z = 0;
    int w = 0;
    int f = 0;
};

template <typename T>
inline cudaChannelFormatDesc cudaCreateChannelDesc() {
    return {};
}

struct cudaResourceDesc {
    int resType = cudaResourceTypeLinear;
    struct {
        struct {
            void* devPtr = nullptr;
            cudaChannelFormatDesc desc{};
            size_t sizeInBytes = 0;
        } linear;
    } res;
};

struct cudaTextureDesc {
    int addressMode[3] = {cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp};
    int filterMode = cudaFilterModePoint;
    int readMode = cudaReadModeElementType;
    int normalizedCoords = 0;
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
    if (!ptr) {
        return cudaErrorInvalidValue;
    }
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
    if (count > 0 && (!dst || !src)) {
        return cudaErrorInvalidValue;
    }
    if (count > 0) {
        std::memcpy(dst, src, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* dst, int value, size_t count) {
    if (count > 0 && !dst) {
        return cudaErrorInvalidValue;
    }
    if (count > 0) {
        std::memset(dst, value, count);
    }
    return cudaSuccess;
}

inline cudaError_t cudaCreateTextureObject(cudaTextureObject_t* tex,
                                           const cudaResourceDesc*,
                                           const cudaTextureDesc*,
                                           const void*) {
    static cudaTextureObject_t next_texture = 1;
    if (!tex) {
        return cudaErrorInvalidValue;
    }
    *tex = next_texture++;
    return cudaSuccess;
}

inline cudaError_t cudaDestroyTextureObject(cudaTextureObject_t) {
    return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    if (!prop) {
        return cudaErrorInvalidValue;
    }
    *prop = {};
    return cudaSuccess;
}

#endif

#endif  // SPMV_CUDA_COMPAT_H
