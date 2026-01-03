#include "spmv/spmv.h"
#include <cstring>

namespace spmv {

void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y) {
    if (!A || !x || !y) return;
    
    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int j = A->row_ptrs[i]; j < A->row_ptrs[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}

void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y) {
    if (!A || !x || !y) return;
    
    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int k = 0; k < A->max_nnz_per_row; k++) {
            int idx = ell_index(i, k, A->num_rows);
            int col = A->col_indices[idx];
            if (col >= 0) {
                sum += A->values[idx] * x[col];
            }
        }
        y[i] = sum;
    }
}

SpMVConfig spmv_auto_config(const CSRMatrix* A) {
    SpMVConfig config;
    config.block_size = 256;
    config.use_texture = (A->num_cols > 10000);
    
    CSRStats stats = csr_compute_stats(A);
    
    if (stats.avg_nnz_per_row < 4.0f) {
        config.kernel_type = SpMVConfig::SCALAR_CSR;
    } else if (stats.skewness < 10.0f) {
        config.kernel_type = SpMVConfig::VECTOR_CSR;
    } else {
        config.kernel_type = SpMVConfig::MERGE_PATH;
    }
    
    return config;
}

} // namespace spmv
