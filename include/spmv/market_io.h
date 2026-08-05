#ifndef SPMV_MARKET_IO_H
#define SPMV_MARKET_IO_H

#include "csr_matrix.h"

namespace spmv {

/**
 * @file market_io.h
 * @brief Matrix Market (NIST) coordinate file reader for CSR.
 */

/**
 * @brief Load a Matrix Market coordinate file into a CSR matrix.
 *
 * Supported headers:
 * `%%MatrixMarket matrix coordinate {real|integer|pattern} {general|symmetric}`.
 *
 * Symmetric matrices are expanded to full storage and duplicate (row, col)
 * entries are summed.  Complex matrices and dense "array" files are rejected
 * with INVALID_FORMAT.
 *
 * @param mat Output matrix (host storage is reallocated; any device mirror is
 *            dropped).
 * @param filename Input .mtx file path.
 * @return 0 on success, negative error code on failure.
 */
int csr_read_matrix_market(CSRMatrix* mat, const char* filename);

}  // namespace spmv

#endif  // SPMV_MARKET_IO_H
