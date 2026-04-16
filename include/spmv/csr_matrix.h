#ifndef SPMV_CSR_MATRIX_H
#define SPMV_CSR_MATRIX_H

#include <cstddef>
#include <vector>

#include "common.h"

namespace spmv {

/**
 * @file csr_matrix.h
 * @brief CSR (Compressed Sparse Row) sparse matrix format.
 *
 * The CSR format stores a sparse matrix using three arrays:
 * - values: Non-zero values stored row by row
 * - col_indices: Column index for each non-zero value
 * - row_ptrs: Index into values/col_indices where each row starts
 *
 * This format is memory-efficient and works well for matrices
 * with moderate row-length variation.
 */

/**
 * @brief CSR (Compressed Sparse Row) sparse matrix structure.
 *
 * Stores both host and device memory pointers with ownership tracking.
 */
struct CSRMatrix {
  int num_rows;  ///< Number of rows in the matrix
  int num_cols;  ///< Number of columns in the matrix
  int nnz;       ///< Total number of non-zero elements

  float* values;     ///< Non-zero values array [nnz]
  int* col_indices;  ///< Column indices array [nnz]
  int* row_ptrs;     ///< Row pointers array [num_rows + 1]

  // GPU device pointers
  float* d_values;     ///< Device memory for values
  int* d_col_indices;  ///< Device memory for column indices
  int* d_row_ptrs;     ///< Device memory for row pointers

  // Memory ownership flags
  bool owns_host_memory;  ///< True if host memory should be freed on destroy
  bool
      owns_device_memory;  ///< True if device memory should be freed on destroy
};

/**
 * @brief Create an empty CSR matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param nnz Number of non-zero elements (preallocate space).
 * @return Pointer to new matrix, or nullptr on invalid input.
 */
CSRMatrix* csr_create(int rows, int cols, int nnz);

/**
 * @brief Destroy a CSR matrix and free all memory.
 *
 * Frees both host and device memory if owned.
 *
 * @param mat Matrix to destroy (may be nullptr).
 */
void csr_destroy(CSRMatrix* mat);

/**
 * @brief Convert a dense matrix to CSR format.
 *
 * @param csr Output CSR matrix (must be pre-created).
 * @param dense Input dense matrix in row-major order [rows * cols].
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return 0 on success, negative error code on failure.
 */
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);

/**
 * @brief Convert a CSR matrix to dense format.
 *
 * @param csr Input CSR matrix.
 * @param dense Output dense matrix [num_rows * num_cols].
 * @return 0 on success, negative error code on failure.
 */
int csr_to_dense(const CSRMatrix* csr, float* dense);

/**
 * @brief Get the value at a specific position.
 *
 * @param mat The CSR matrix.
 * @param row Row index (0-indexed).
 * @param col Column index (0-indexed).
 * @return The value at (row, col), or 0.0f if out of bounds or not stored.
 */
float csr_get_element(const CSRMatrix* mat, int row, int col);

/**
 * @brief Copy matrix data to GPU memory.
 *
 * Allocates device memory and copies host data to device.
 * Any existing device memory is freed first.
 *
 * @param mat Matrix to upload.
 * @return 0 on success, negative error code on failure.
 */
int csr_to_gpu(CSRMatrix* mat);

/**
 * @brief Copy matrix data from GPU to host memory.
 *
 * @param mat Matrix with device data to download.
 * @return 0 on success, negative error code on failure.
 */
int csr_from_gpu(CSRMatrix* mat);

/**
 * @brief Free GPU memory associated with the matrix.
 *
 * @param mat Matrix whose device memory should be freed.
 */
void csr_free_gpu(CSRMatrix* mat);

/**
 * @brief Serialize matrix to binary file.
 *
 * File format includes magic number, version, and checksum for integrity.
 *
 * @param mat Matrix to serialize.
 * @param filename Output file path.
 * @return 0 on success, negative error code on failure.
 */
int csr_serialize(const CSRMatrix* mat, const char* filename);

/**
 * @brief Deserialize matrix from binary file.
 *
 * Validates magic number, version, and checksum.
 *
 * @param mat Output matrix (will be reallocated).
 * @param filename Input file path.
 * @return 0 on success, negative error code on failure.
 */
int csr_deserialize(CSRMatrix* mat, const char* filename);

/**
 * @brief Statistics about row lengths in a CSR matrix.
 */
struct CSRStats {
  float avg_nnz_per_row;  ///< Average non-zeros per row
  int max_nnz_per_row;    ///< Maximum non-zeros in any row
  int min_nnz_per_row;    ///< Minimum non-zeros in any row
  float skewness;         ///< Ratio: max / (min + 1), measures irregularity
};

/**
 * @brief Compute statistics about the matrix structure.
 *
 * @param mat The CSR matrix.
 * @return Statistics structure.
 */
CSRStats csr_compute_stats(const CSRMatrix* mat);

/**
 * @brief Validate matrix structure integrity.
 *
 * Checks:
 * - Dimensions are non-negative
 * - Pointers are valid
 * - row_ptrs is monotonically increasing
 * - row_ptrs[0] == 0 and row_ptrs[num_rows] == nnz
 * - All column indices are in valid range [0, num_cols)
 *
 * @param mat Matrix to validate.
 * @return true if structure is valid, false otherwise.
 */
bool csr_validate(const CSRMatrix* mat);

}  // namespace spmv

#endif  // SPMV_CSR_MATRIX_H
