#include "spmv/spmv.h"

namespace spmv {

namespace {

int no_cuda_error() {
    return static_cast<int>(SpMVError::KERNEL_LAUNCH);
}

}  // namespace

SpMVResult spmv_csr(const CSRMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    SpMVExecutionContext*) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = no_cuda_error();
    return result;
}

SpMVResult spmv_ell(const ELLMatrix*, const float*, float* d_y, const SpMVConfig*, int,
                    SpMVExecutionContext*) {
    SpMVResult result;
    result.y = d_y;
    result.error_code = no_cuda_error();
    return result;
}

}  // namespace spmv
