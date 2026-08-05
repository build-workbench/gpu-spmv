#ifndef SPMV_INTERNAL_KERNEL_SELECTOR_H
#define SPMV_INTERNAL_KERNEL_SELECTOR_H

#include "spmv/spmv.h"

namespace spmv {

/**
 * @brief Pure kernel selection logic.
 *
 * All policy decisions are concentrated here.  The function takes only
 * immutable inputs (stats, thresholds) and returns a deterministic
 * SpMVConfig.  No global state, no matrix internals.
 */
SpMVConfig select_kernel(const CSRStats& stats, const SpMVThresholds& thresholds);

}  // namespace spmv

#endif  // SPMV_INTERNAL_KERNEL_SELECTOR_H
