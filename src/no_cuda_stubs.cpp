#include "spmv/spmv.h"

namespace spmv {

SpMVResult spmv_csr(const CSRMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    cudaStream_t) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
    return result;
}

SpMVResult spmv_ell(const ELLMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    cudaStream_t) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
    return result;
}

}  // namespace spmv
