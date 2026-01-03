#ifndef SPMV_CUDA_BUFFER_H
#define SPMV_CUDA_BUFFER_H

#include "common.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace spmv {

// RAII 风格的 GPU 内存管理
template<typename T>
class CudaBuffer {
public:
    CudaBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit CudaBuffer(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw CudaException(err);
            }
        }
    }
    
    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
    
    // 禁止拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // 允许移动
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return ptr_ == nullptr || size_ == 0; }
    
    // 从主机复制数据到设备
    void copyFromHost(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        CUDA_CHECK_THROW(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // 从设备复制数据到主机
    void copyToHost(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        CUDA_CHECK_THROW(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    // 重新分配
    void resize(size_t new_count) {
        if (new_count == size_) return;
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = new_count;
        if (new_count > 0) {
            CUDA_CHECK_THROW(cudaMalloc(&ptr_, new_count * sizeof(T)));
        }
    }
    
    // 释放内存
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }
    
private:
    T* ptr_;
    size_t size_;
};

} // namespace spmv

#endif // SPMV_CUDA_BUFFER_H
