#ifndef SPMV_INTERNAL_TEXTURE_CACHE_H
#define SPMV_INTERNAL_TEXTURE_CACHE_H

#include "spmv/spmv.h"

namespace spmv {

int spmv_prepare_texture(SpMVExecutionContext* context, const float* d_x, size_t x_length,
                         bool requested, cudaTextureObject_t* tex_out, bool* use_texture_out);

}  // namespace spmv

#endif  // SPMV_INTERNAL_TEXTURE_CACHE_H
