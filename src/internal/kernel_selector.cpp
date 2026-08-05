#include "kernel_selector.h"

namespace spmv {

SpMVConfig select_kernel(const CSRStats& stats, const SpMVThresholds& thresholds) {
    SpMVConfig config(SpMVConfig::SCALAR_CSR, DEFAULT_BLOCK_SIZE);

    if (stats.avg_nnz_per_row < thresholds.avg_nnz_threshold) {
        config.kernel_type = SpMVConfig::SCALAR_CSR;
    } else if (stats.skewness < thresholds.skewness_threshold) {
        config.kernel_type = SpMVConfig::VECTOR_CSR;
    } else {
        config.kernel_type = SpMVConfig::MERGE_PATH;
    }

    return config;
}

}  // namespace spmv
