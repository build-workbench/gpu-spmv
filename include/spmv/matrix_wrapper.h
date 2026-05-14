#ifndef SPMV_MATRIX_WRAPPER_H
#define SPMV_MATRIX_WRAPPER_H

#include <utility>

#include "csr_matrix.h"
#include "ell_matrix.h"

namespace spmv {

/**
 * @file matrix_wrapper.h
 * @brief RAII wrappers for sparse matrix structures.
 *
 * Provides automatic memory management for CSR and ELL matrices.
 * These wrappers ensure proper cleanup of both host and device memory.
 */

/**
 * @brief RAII wrapper for CSRMatrix.
 *
 * Automatically manages host and device memory lifecycle.
 * Move-only to prevent accidental double-frees.
 *
 * Example usage:
 * @code
 * {
 *     CSRMatrixWrapper mat(100, 100, 500);
 *     mat.from_dense(dense_data);
 *     mat.to_gpu();
 *     spmv_csr(mat.get(), d_x, d_y, nullptr);
 *     // Memory automatically freed when mat goes out of scope
 * }
 * @endcode
 */
class CSRMatrixWrapper {
   public:
    /**
     * @brief Construct an empty wrapper.
     */
    CSRMatrixWrapper() : mat_(nullptr) {}

    /**
     * @brief Construct and allocate a matrix.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param nnz Number of non-zeros (preallocation).
     */
    CSRMatrixWrapper(int rows, int cols, int nnz) : mat_(csr_create(rows, cols, nnz)) {}

    /**
     * @brief Take ownership of an existing matrix.
     * @param mat Existing matrix (ownership transferred).
     */
    explicit CSRMatrixWrapper(CSRMatrix* mat) : mat_(mat) {}

    /**
     * @brief Destructor - frees all memory.
     */
    ~CSRMatrixWrapper() {
        if (mat_) {
            csr_destroy(mat_);
            mat_ = nullptr;
        }
    }

    // Non-copyable
    CSRMatrixWrapper(const CSRMatrixWrapper&) = delete;
    CSRMatrixWrapper& operator=(const CSRMatrixWrapper&) = delete;

    /**
     * @brief Move constructor.
     */
    CSRMatrixWrapper(CSRMatrixWrapper&& other) noexcept : mat_(other.mat_) { other.mat_ = nullptr; }

    /**
     * @brief Move assignment operator.
     */
    CSRMatrixWrapper& operator=(CSRMatrixWrapper&& other) noexcept {
        if (this != &other) {
            if (mat_) {
                csr_destroy(mat_);
            }
            mat_ = other.mat_;
            other.mat_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get raw matrix pointer.
     * @return Pointer to wrapped matrix.
     */
    CSRMatrix* get() { return mat_; }

    /**
     * @brief Get const raw matrix pointer.
     * @return Const pointer to wrapped matrix.
     */
    const CSRMatrix* get() const { return mat_; }

    /**
     * @brief Access matrix members.
     * @return Pointer to matrix.
     */
    CSRMatrix* operator->() { return mat_; }

    /**
     * @brief Access matrix members (const).
     * @return Const pointer to matrix.
     */
    const CSRMatrix* operator->() const { return mat_; }

    /**
     * @brief Check if wrapper holds a matrix.
     * @return true if matrix is valid.
     */
    bool valid() const { return mat_ != nullptr; }

    /**
     * @brief Check if matrix has device data.
     * @return true if device memory is allocated.
     */
    bool has_device_data() const { return mat_ && csr_has_device_data(mat_); }

    /**
     * @brief Convert from dense matrix.
     * @param dense Dense matrix data [rows * cols].
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return 0 on success, negative on error.
     */
    int from_dense(const float* dense, int rows, int cols) {
        if (!mat_) {
            mat_ = csr_create(rows, cols, 0);
            if (!mat_)
                return static_cast<int>(SpMVError::OUT_OF_MEMORY);
        }
        return csr_from_dense(mat_, dense, rows, cols);
    }

    /**
     * @brief Convert to dense matrix.
     * @param dense Output buffer [num_rows * num_cols].
     * @return 0 on success, negative on error.
     */
    int to_dense(float* dense) const {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return csr_to_dense(mat_, dense);
    }

    /**
     * @brief Upload data to GPU.
     * @return 0 on success, negative on error.
     */
    int to_gpu() {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return csr_to_gpu(mat_);
    }

    /**
     * @brief Download data from GPU.
     * @return 0 on success, negative on error.
     */
    int from_gpu() {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return csr_from_gpu(mat_);
    }

    /**
     * @brief Serialize to file.
     * @param filename Output file path.
     * @return 0 on success, negative on error.
     */
    int serialize(const char* filename) const {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return csr_serialize(mat_, filename);
    }

    /**
     * @brief Deserialize from file.
     * @param filename Input file path.
     * @return 0 on success, negative on error.
     */
    int deserialize(const char* filename) {
        if (!mat_) {
            mat_ = csr_create(0, 0, 0);
            if (!mat_)
                return static_cast<int>(SpMVError::OUT_OF_MEMORY);
        }
        return csr_deserialize(mat_, filename);
    }

    /**
     * @brief Validate matrix structure.
     * @return true if structure is valid.
     */
    bool validate() const {
        if (!mat_)
            return false;
        return csr_validate(mat_);
    }

    /**
     * @brief Compute matrix statistics.
     * @return Statistics structure.
     */
    CSRStats compute_stats() const {
        if (!mat_)
            return CSRStats{};
        return csr_compute_stats(mat_);
    }

    /**
     * @brief Release ownership of the matrix.
     *
     * Caller is responsible for freeing the returned matrix.
     * @return Pointer to matrix (caller owns).
     */
    CSRMatrix* release() {
        CSRMatrix* tmp = mat_;
        mat_ = nullptr;
        return tmp;
    }

    /**
     * @brief Reset wrapper, freeing the held matrix.
     */
    void reset() {
        if (mat_) {
            csr_destroy(mat_);
            mat_ = nullptr;
        }
    }

   private:
    CSRMatrix* mat_;
};

/**
 * @brief RAII wrapper for ELLMatrix.
 *
 * Automatically manages host and device memory lifecycle.
 * Move-only to prevent accidental double-frees.
 */
class ELLMatrixWrapper {
   public:
    /**
     * @brief Construct an empty wrapper.
     */
    ELLMatrixWrapper() : mat_(nullptr) {}

    /**
     * @brief Construct and allocate a matrix.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param max_nnz_per_row Maximum non-zeros per row.
     */
    ELLMatrixWrapper(int rows, int cols, int max_nnz_per_row)
        : mat_(ell_create(rows, cols, max_nnz_per_row)) {}

    /**
     * @brief Take ownership of an existing matrix.
     * @param mat Existing matrix (ownership transferred).
     */
    explicit ELLMatrixWrapper(ELLMatrix* mat) : mat_(mat) {}

    /**
     * @brief Destructor - frees all memory.
     */
    ~ELLMatrixWrapper() {
        if (mat_) {
            ell_destroy(mat_);
            mat_ = nullptr;
        }
    }

    // Non-copyable
    ELLMatrixWrapper(const ELLMatrixWrapper&) = delete;
    ELLMatrixWrapper& operator=(const ELLMatrixWrapper&) = delete;

    /**
     * @brief Move constructor.
     */
    ELLMatrixWrapper(ELLMatrixWrapper&& other) noexcept : mat_(other.mat_) { other.mat_ = nullptr; }

    /**
     * @brief Move assignment operator.
     */
    ELLMatrixWrapper& operator=(ELLMatrixWrapper&& other) noexcept {
        if (this != &other) {
            if (mat_) {
                ell_destroy(mat_);
            }
            mat_ = other.mat_;
            other.mat_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get raw matrix pointer.
     * @return Pointer to wrapped matrix.
     */
    ELLMatrix* get() { return mat_; }

    /**
     * @brief Get const raw matrix pointer.
     * @return Const pointer to wrapped matrix.
     */
    const ELLMatrix* get() const { return mat_; }

    /**
     * @brief Access matrix members.
     * @return Pointer to matrix.
     */
    ELLMatrix* operator->() { return mat_; }

    /**
     * @brief Access matrix members (const).
     * @return Const pointer to matrix.
     */
    const ELLMatrix* operator->() const { return mat_; }

    /**
     * @brief Check if wrapper holds a matrix.
     * @return true if matrix is valid.
     */
    bool valid() const { return mat_ != nullptr; }

    /**
     * @brief Check if matrix has device data.
     * @return true if device memory is allocated.
     */
    bool has_device_data() const { return mat_ && ell_has_device_data(mat_); }

    /**
     * @brief Convert from dense matrix.
     * @param dense Dense matrix data [rows * cols].
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return 0 on success, negative on error.
     */
    int from_dense(const float* dense, int rows, int cols) {
        if (!mat_) {
            mat_ = ell_create(rows, cols, 0);
            if (!mat_)
                return static_cast<int>(SpMVError::OUT_OF_MEMORY);
        }
        return ell_from_dense(mat_, dense, rows, cols);
    }

    /**
     * @brief Convert from CSR matrix.
     * @param csr Source CSR matrix.
     * @return 0 on success, negative on error.
     */
    int from_csr(const CSRMatrix* csr) {
        if (!mat_ || !csr)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return ell_from_csr(mat_, csr);
    }

    /**
     * @brief Convert to dense matrix.
     * @param dense Output buffer [num_rows * num_cols].
     * @return 0 on success, negative on error.
     */
    int to_dense(float* dense) const {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return ell_to_dense(mat_, dense);
    }

    /**
     * @brief Upload data to GPU.
     * @return 0 on success, negative on error.
     */
    int to_gpu() {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return ell_to_gpu(mat_);
    }

    /**
     * @brief Download data from GPU.
     * @return 0 on success, negative on error.
     */
    int from_gpu() {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return ell_from_gpu(mat_);
    }

    /**
     * @brief Serialize to file.
     * @param filename Output file path.
     * @return 0 on success, negative on error.
     */
    int serialize(const char* filename) const {
        if (!mat_)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return ell_serialize(mat_, filename);
    }

    /**
     * @brief Deserialize from file.
     * @param filename Input file path.
     * @return 0 on success, negative on error.
     */
    int deserialize(const char* filename) {
        if (!mat_) {
            mat_ = ell_create(0, 0, 0);
            if (!mat_)
                return static_cast<int>(SpMVError::OUT_OF_MEMORY);
        }
        return ell_deserialize(mat_, filename);
    }

    /**
     * @brief Validate matrix structure.
     * @return true if structure is valid.
     */
    bool validate() const {
        if (!mat_)
            return false;
        return ell_validate(mat_);
    }

    /**
     * @brief Release ownership of the matrix.
     *
     * Caller is responsible for freeing the returned matrix.
     * @return Pointer to matrix (caller owns).
     */
    ELLMatrix* release() {
        ELLMatrix* tmp = mat_;
        mat_ = nullptr;
        return tmp;
    }

    /**
     * @brief Reset wrapper, freeing the held matrix.
     */
    void reset() {
        if (mat_) {
            ell_destroy(mat_);
            mat_ = nullptr;
        }
    }

   private:
    ELLMatrix* mat_;
};

}  // namespace spmv

#endif  // SPMV_MATRIX_WRAPPER_H
