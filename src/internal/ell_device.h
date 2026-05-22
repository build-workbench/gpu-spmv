#ifndef SPMV_INTERNAL_ELL_DEVICE_H
#define SPMV_INTERNAL_ELL_DEVICE_H

#include "spmv/ell_matrix.h"

namespace spmv {

// Internal device-state accessors for ELLMatrix.
// These are NOT part of the public API.

void* ell_create_device_state();
void ell_destroy_device_state(ELLMatrix* mat);

float* ell_d_values(ELLMatrix* mat);
const float* ell_d_values(const ELLMatrix* mat);
int* ell_d_col_indices(ELLMatrix* mat);
const int* ell_d_col_indices(const ELLMatrix* mat);

bool ell_has_device_data(const ELLMatrix* mat);

// Free only device memory (called internally by to_gpu / destroy).
void ell_free_device_data(ELLMatrix* mat);
int ell_upload_device_data(ELLMatrix* mat);
int ell_download_device_data(ELLMatrix* mat);

}  // namespace spmv

#endif  // SPMV_INTERNAL_ELL_DEVICE_H
