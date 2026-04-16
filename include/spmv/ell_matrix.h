#ifndef SPMV_ELL_MATRIX_H
#define SPMV_ELL_MATRIX_H

#include <cstddef>

#include "common.h"
#include "csr_matrix.h"

namespace spmv {

/**
 * @file ell_matrix.h
 * @brief ELL (ELLPACK) sparse matrix format.
 *
 * The ELL format stores a sparse matrix in column-major order
 * for coalesced GPU memory access. Each row is padded to the
 * same length (max_nnz_per_row), with -1 indicating padding.
 *
 * Best suited for matrices with uniform row lengths.
 */

/**
 * @brief ELL (ELLPACK) sparse matrix structure.
 *
 * Uses column-major storage for GPU memory coalescing.
 */
struct ELLMatrix {
    int num_rows;         ///< Number of rows
    int num_cols;         ///< Number of columns
    int max_nnz_per_row;  ///< Maximum non-zeros in any row (determines padding)
    int nnz;              ///< Actual total non-zero count

    /// Column-major storage: values[k * num_rows + row]
    float* values;     ///< Values array [num_rows * max_nnz_per_row]
    int* col_indices;  ///< Column indices [num_rows * max_nnz_per_row], -1 = padding

    // GPU device pointers
    float* d_values;     ///< Device memory for values
    int* d_col_indices;  ///< Device memory for column indices

    // Memory ownership flags
    bool owns_host_memory;    ///< True if host memory should be freed on destroy
    bool owns_device_memory;  ///< True if device memory should be freed on destroy
};

/**
 * @brief Create an empty ELL matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param max_nnz_per_row Maximum non-zeros per row (determines storage).
 * @return Pointer to new matrix, or nullptr on invalid input.
 */
ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row);

/**
 * @brief Destroy an ELL matrix and free all memory.
 *
 * @param mat Matrix to destroy (may be nullptr).
 */
void ell_destroy(ELLMatrix* mat);

/**
 * @brief Convert a dense matrix to ELL format.
 *
 * @param ell Output ELL matrix.
 * @param dense Input dense matrix in row-major order.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return 0 on success, negative error code on failure.
 */
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);

/**
 * @brief Convert a CSR matrix to ELL format.
 *
 * @param ell Output ELL matrix.
 * @param csr Input CSR matrix.
 * @return 0 on success, negative error code on failure.
 */
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);

/**
 * @brief Convert an ELL matrix to dense format.
 *
 * @param ell Input ELL matrix.
 * @param dense Output dense matrix.
 * @return 0 on success, negative error code on failure.
 */
int ell_to_dense(const ELLMatrix* ell, float* dense);

/**
 * @brief Get the value at a specific position.
 *
 * @param mat The ELL matrix.
 * @param row Row index.
 * @param col Column index.
 * @return Value at (row, col), or 0.0f if not found or out of bounds.
 */
float ell_get_element(const ELLMatrix* mat, int row, int col);

/**
 * @brief Copy matrix data to GPU memory.
 *
 * @param mat Matrix to upload.
 * @return 0 on success, negative error code on failure.
 */
int ell_to_gpu(ELLMatrix* mat);

/**
 * @brief Copy matrix data from GPU to host memory.
 *
 * @param mat Matrix with device data.
 * @return 0 on success, negative error code on failure.
 */
int ell_from_gpu(ELLMatrix* mat);

/**
 * @brief Free GPU memory associated with the matrix.
 *
 * @param mat Matrix whose device memory should be freed.
 */
void ell_free_gpu(ELLMatrix* mat);

/**
 * @brief Serialize matrix to binary file.
 *
 * @param mat Matrix to serialize.
 * @param filename Output file path.
 * @return 0 on success, negative error code on failure.
 */
int ell_serialize(const ELLMatrix* mat, const char* filename);

/**
 * @brief Deserialize matrix from binary file.
 *
 * @param mat Output matrix.
 * @param filename Input file path.
 * @return 0 on success, negative error code on failure.
 */
int ell_deserialize(ELLMatrix* mat, const char* filename);

/**
 * @brief Compute column-major index for ELL storage.
 *
 * @param row Row index.
 * @param k Index within the row (0 to max_nnz_per_row-1).
 * @param num_rows Total number of rows.
 * @return Linear index into values/col_indices arrays.
 */
inline int ell_index(int row, int k, int num_rows) {
    return k * num_rows + row;
}

/**
 * @brief Validate matrix structure integrity.
 *
 * Checks dimensions, pointers, and that all column indices
 * are in valid range [-1, num_cols) where -1 indicates padding.
 *
 * @param mat Matrix to validate.
 * @return true if structure is valid, false otherwise.
 */
bool ell_validate(const ELLMatrix* mat);

}  // namespace spmv

#endif  // SPMV_ELL_MATRIX_H
