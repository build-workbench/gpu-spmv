#ifndef SPMV_CUDA_BUFFER_H
#define SPMV_CUDA_BUFFER_H

#include <cstddef>
#include <utility>
#include <vector>

#include "common.h"
#include "cuda_compat.h"

namespace spmv {

/**
 * @file cuda_buffer.h
 * @brief RAII-style GPU memory management.
 *
 * Provides a template class for automatic GPU memory management
 * following RAII principles. Memory is automatically freed when
 * the buffer goes out of scope.
 */

/**
 * @brief RAII wrapper for GPU memory.
 *
 * @tparam T Element type.
 *
 * This class manages GPU memory with automatic cleanup.
 * It is move-only (not copyable) to prevent accidental double-frees.
 *
 * Example usage:
 * @code
 * CudaBuffer<float> buf(1000);  // Allocates 1000 floats
 * buf.copyFromHost(host_data, 1000);
 * // Use buf.get() to get device pointer
 * // Memory freed automatically when buf goes out of scope
 * @endcode
 */
template <typename T>
class CudaBuffer {
   public:
    /**
     * @brief Construct an empty buffer.
     */
    CudaBuffer() : ptr_(nullptr), size_(0) {}

    /**
     * @brief Construct a buffer with given capacity.
     *
     * @param count Number of elements to allocate.
     * @throws CudaException if allocation fails.
     */
    explicit CudaBuffer(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw CudaException(err);
            }
        }
    }

    /**
     * @brief Destructor - frees GPU memory.
     */
    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    // Non-copyable
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    /**
     * @brief Move constructor.
     */
    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator.
     */
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw device pointer.
     * @return Device pointer (may be nullptr).
     */
    T* get() { return ptr_; }

    /**
     * @brief Get const raw device pointer.
     * @return Const device pointer.
     */
    const T* get() const { return ptr_; }

    /**
     * @brief Get number of elements.
     * @return Element count.
     */
    size_t size() const { return size_; }

    /**
     * @brief Get buffer size in bytes.
     * @return Byte count.
     */
    size_t bytes() const { return size_ * sizeof(T); }

    /**
     * @brief Check if buffer is empty.
     * @return true if no memory allocated.
     */
    bool empty() const { return ptr_ == nullptr || size_ == 0; }

    /**
     * @brief Fill buffer with a byte value.
     * @param value Byte value to fill (default 0).
     */
    void memset(int value = 0) {
        if (ptr_ && size_ > 0) {
            SPMV_CUDA_CHECK_THROW(cudaMemset(ptr_, value, size_ * sizeof(T)));
        }
    }

    /**
     * @brief Fill buffer with a value.
     * @param value Value to fill all elements with.
     */
    void fill(const T& value) {
        if (!ptr_ || size_ == 0)
            return;
        std::vector<T> host_data(size_, value);
        SPMV_CUDA_CHECK_THROW(
            cudaMemcpy(ptr_, host_data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy data from host to device.
     *
     * @param host_data Source host pointer.
     * @param count Number of elements to copy.
     * @throws std::runtime_error if count exceeds buffer size.
     */
    void copyFromHost(const T* host_data, size_t count) {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        SPMV_CUDA_CHECK_THROW(
            cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy data from device to host.
     *
     * @param host_data Destination host pointer.
     * @param count Number of elements to copy.
     * @throws std::runtime_error if count exceeds buffer size.
     */
    void copyToHost(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Copy size exceeds buffer size");
        }
        SPMV_CUDA_CHECK_THROW(
            cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Resize buffer (reallocates if necessary).
     *
     * Like std::vector::resize, the min(old, new) leading elements are
     * preserved; any additional elements are left uninitialized.
     *
     * @param new_count New element count.
     * @throws CudaException if allocation or the device-to-device copy fails.
     */
    void resize(size_t new_count) {
        if (new_count == size_)
            return;
        if (new_count == 0) {
            release();
            return;
        }
        T* new_ptr = nullptr;
        SPMV_CUDA_CHECK_THROW(cudaMalloc(&new_ptr, new_count * sizeof(T)));
        if (ptr_ && size_ > 0) {
            size_t copy_count = (size_ < new_count) ? size_ : new_count;
            cudaError_t err =
                cudaMemcpy(new_ptr, ptr_, copy_count * sizeof(T), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                // Leave the original buffer intact on failure.
                cudaFree(new_ptr);
                throw CudaException(err);
            }
        }
        cudaFree(ptr_);
        ptr_ = new_ptr;
        size_ = new_count;
    }

    /**
     * @brief Release memory and reset to empty.
     */
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

   private:
    T* ptr_;       ///< Device pointer
    size_t size_;  ///< Element count
};

}  // namespace spmv

#endif  // SPMV_CUDA_BUFFER_H
