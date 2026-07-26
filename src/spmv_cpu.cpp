#include "spmv/spmv.h"

#include <cstring>
#include <mutex>

#include "internal/kernel_selector.h"

namespace spmv {

static std::mutex g_thresholds_mutex;
static SpMVThresholds g_thresholds;

SpMVThresholds spmv_get_thresholds() {
    std::lock_guard<std::mutex> lock(g_thresholds_mutex);
    return g_thresholds;
}

void spmv_set_thresholds(const SpMVThresholds& thresholds) {
    std::lock_guard<std::mutex> lock(g_thresholds_mutex);
    g_thresholds = thresholds;
}

void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y) {
    if (!A || !x || !y)
        return;

    for (int i = 0; i < A->num_rows; i++) {
        float sum = 0.0f;
        for (int j = A->row_ptrs[i]; j < A->row_ptrs[i + 1]; j++) {
            sum += A->values[j] * x[A->col_indices[j]];
        }
        y[i] = sum;
    }
}

void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y) {
    if (!A || !x || !y)
        return;

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
    if (!A || A->num_rows < 0 || A->num_cols < 0 || A->nnz < 0 || !A->row_ptrs) {
        return SpMVConfig(SpMVConfig::SCALAR_CSR, DEFAULT_BLOCK_SIZE);
    }

    if (A->num_rows == 0 || A->num_cols == 0 || A->nnz == 0) {
        return SpMVConfig(SpMVConfig::SCALAR_CSR, DEFAULT_BLOCK_SIZE);
    }

    CSRStats stats = csr_compute_stats(A);
    SpMVThresholds thresholds;
    {
        std::lock_guard<std::mutex> lock(g_thresholds_mutex);
        thresholds = g_thresholds;
    }
    return select_kernel(stats, A->num_cols, thresholds);
}

}  // namespace spmv
