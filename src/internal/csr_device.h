#ifndef SPMV_INTERNAL_CSR_DEVICE_H
#define SPMV_INTERNAL_CSR_DEVICE_H

#include "spmv/csr_matrix.h"

namespace spmv {

// Internal device-state accessors for CSRMatrix.
// These are NOT part of the public API; they are exposed only to
// compilation units inside src/ that need to launch kernels.

void* csr_create_device_state();
void csr_destroy_device_state(CSRMatrix* mat);

float* csr_d_values(CSRMatrix* mat);
const float* csr_d_values(const CSRMatrix* mat);
int* csr_d_col_indices(CSRMatrix* mat);
const int* csr_d_col_indices(const CSRMatrix* mat);
int* csr_d_row_ptrs(CSRMatrix* mat);
const int* csr_d_row_ptrs(const CSRMatrix* mat);

bool csr_has_device_data(const CSRMatrix* mat);

// Free only device memory (called internally by to_gpu / destroy).
void csr_free_device_data(CSRMatrix* mat);
int csr_upload_device_data(CSRMatrix* mat);
int csr_download_device_data(CSRMatrix* mat);

}  // namespace spmv

#endif  // SPMV_INTERNAL_CSR_DEVICE_H
