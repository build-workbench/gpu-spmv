#include "internal/texture_cache.h"

namespace spmv {

int spmv_prepare_texture(SpMVExecutionContext* context, const float* d_x, size_t x_length,
                         bool requested, cudaTextureObject_t* tex_out, bool* use_texture_out) {
    if (!tex_out || !use_texture_out) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    *tex_out = 0;
    *use_texture_out = false;

    if (!requested || !d_x || x_length == 0) {
        if (context) {
            context->reset();
        }
        return static_cast<int>(SpMVError::SUCCESS);
    }

    if (!context) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    bool needs_rebuild = !context->texture_enabled_ || context->tex_x_ == 0 || context->cached_x_ != d_x ||
                         context->cached_x_length_ != x_length;
    if (needs_rebuild) {
        context->reset();

        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = const_cast<float*>(d_x);
        res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
        res_desc.res.linear.sizeInBytes = x_length * sizeof(float);

        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModePoint;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaError_t err = cudaCreateTextureObject(&context->tex_x_, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) {
            context->reset();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        context->cached_x_ = d_x;
        context->cached_x_length_ = x_length;
        context->texture_enabled_ = true;
    }

    *tex_out = context->tex_x_;
    *use_texture_out = true;
    return static_cast<int>(SpMVError::SUCCESS);
}

}  // namespace spmv
