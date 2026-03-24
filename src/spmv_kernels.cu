#include "spmv/spmv.h"
#include "spmv/bandwidth.h"
#include <cuda_runtime.h>
#include <chrono>

namespace spmv {

// ---------- RAII helpers ----------

// RAII wrapper for CUDA events (timing)
struct CudaTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t status = cudaSuccess;

    CudaTimer() {
        status = cudaEventCreate(&start);
        if (status != cudaSuccess) {
            start = nullptr;
            return;
        }

        status = cudaEventCreate(&stop);
        if (status != cudaSuccess) {
            cudaEventDestroy(start);
            start = nullptr;
            stop = nullptr;
        }
    }

    ~CudaTimer() {
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    cudaError_t init_status() const { return status; }

    cudaError_t record_start() {
        return (status == cudaSuccess) ? cudaEventRecord(start) : status;
    }

    cudaError_t record_stop() {
        if (status != cudaSuccess) {
            return status;
        }

        cudaError_t err = cudaEventRecord(stop);
        if (err != cudaSuccess) {
            return err;
        }
        return cudaEventSynchronize(stop);
    }

    cudaError_t elapsed_ms(float* ms) const {
        if (!ms) {
            return cudaErrorInvalidValue;
        }
        if (status != cudaSuccess) {
            return status;
        }
        return cudaEventElapsedTime(ms, start, stop);
    }
};

static int map_cuda_error(cudaError_t err, SpMVError fallback) {
    if (err == cudaSuccess) {
        return static_cast<int>(SpMVError::SUCCESS);
    }
    if (err == cudaErrorMemoryAllocation) {
        return static_cast<int>(SpMVError::CUDA_MALLOC);
    }
    return static_cast<int>(fallback);
}

// RAII wrapper for a CUDA texture object
struct ScopedTexture {
    cudaTextureObject_t tex = 0;
    bool valid = false;

    ScopedTexture() = default;
    ~ScopedTexture() { reset(); }
    ScopedTexture(const ScopedTexture&) = delete;
    ScopedTexture& operator=(const ScopedTexture&) = delete;

    void reset() {
        if (valid) {
            cudaDestroyTextureObject(tex);
            tex = 0;
            valid = false;
        }
    }

    // Returns SpMVError code
    int create(const float* d_x, size_t count) {
        reset();
        if (!d_x || count == 0) return static_cast<int>(SpMVError::INVALID_ARGUMENT);

        cudaResourceDesc res_desc{};
        res_desc.resType = cudaResourceTypeLinear;
        res_desc.res.linear.devPtr = const_cast<float*>(d_x);
        res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
        res_desc.res.linear.sizeInBytes = count * sizeof(float);

        cudaTextureDesc tex_desc{};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModePoint;
        tex_desc.readMode = cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;

        cudaError_t err = cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) return static_cast<int>(SpMVError::CUDA_MALLOC);
        valid = true;
        return static_cast<int>(SpMVError::SUCCESS);
    }
};

static int prepare_texture_context(SpMVExecutionContext* context,
                                   const float* d_x,
                                   size_t x_length,
                                   bool requested_texture,
                                   cudaTextureObject_t* tex_out,
                                   bool* use_texture_out) {
    if (!tex_out || !use_texture_out) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    *tex_out = 0;
    *use_texture_out = false;

    if (!requested_texture || !d_x || x_length == 0) {
        if (context) {
            context->reset();
        }
        return static_cast<int>(SpMVError::SUCCESS);
    }

    if (!context) {
        return static_cast<int>(SpMVError::SUCCESS);
    }

    bool needs_rebuild = !context->texture_enabled || context->tex_x == 0 ||
                         context->cached_x != d_x || context->cached_x_length != x_length;
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

        cudaError_t err = cudaCreateTextureObject(&context->tex_x, &res_desc, &tex_desc, nullptr);
        if (err != cudaSuccess) {
            context->reset();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        context->cached_x = d_x;
        context->cached_x_length = x_length;
        context->texture_enabled = true;
    }

    *tex_out = context->tex_x;
    *use_texture_out = true;
    return static_cast<int>(SpMVError::SUCCESS);
}

__device__ __forceinline__ float fetch_x(const float* x,
                                         cudaTextureObject_t tex_x,
                                         bool use_texture,
                                         int idx) {
    return use_texture ? tex1Dfetch<float>(tex_x, idx) : x[idx];
}

// Merge Path 辅助结构
struct MergeCoordinate {
    int row;
    int nz;
};

// Merge Path 搜索
__device__ MergeCoordinate merge_path_search(
    int diagonal,
    const int* row_ptrs,
    int num_rows,
    int nnz
) {
    int x_min = max(diagonal - nnz, 0);
    int x_max = min(diagonal, num_rows);

    while (x_min < x_max) {
        int x_mid = (x_min + x_max) / 2;
        int y_mid = diagonal - x_mid;

        if (row_ptrs[x_mid] <= y_mid) {
            x_min = x_mid + 1;
        } else {
            x_max = x_mid;
        }
    }

    MergeCoordinate coord;
    coord.row = x_min;
    coord.nz = diagonal - x_min;
    return coord;
}

// Merge Path Kernel
__global__ void spmv_csr_merge_path_kernel(
    int num_rows,
    int nnz,
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* x,
    cudaTextureObject_t tex_x,
    bool use_texture,
    float* y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = num_rows + nnz;

    // 每个线程处理的工作量
    int work_per_thread = (total_work + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);

    int diagonal_start = tid * work_per_thread;
    int diagonal_end = min(diagonal_start + work_per_thread, total_work);

    if (diagonal_start >= total_work) return;

    MergeCoordinate start = merge_path_search(diagonal_start, row_ptrs, num_rows, nnz);
    MergeCoordinate end = merge_path_search(diagonal_end, row_ptrs, num_rows, nnz);

    // 处理分配的工作
    int current_row = start.row;
    int current_nz = start.nz;
    float sum = 0.0f;

    while (current_row < end.row || (current_row == end.row && current_nz < end.nz)) {
        if (current_row < num_rows) {
            int row_end = row_ptrs[current_row + 1];

            while (current_nz < row_end && (current_row < end.row || current_nz < end.nz)) {
                sum += values[current_nz] * fetch_x(x, tex_x, use_texture, col_indices[current_nz]);
                current_nz++;
            }

            if (current_nz == row_end) {
                atomicAdd(&y[current_row], sum);
                sum = 0.0f;
                current_row++;
                current_nz = (current_row < num_rows) ? row_ptrs[current_row] : nnz;
            }
        } else {
            break;
        }
    }

    // 处理剩余的部分和
    if (sum != 0.0f && current_row < num_rows) {
        atomicAdd(&y[current_row], sum);
    }
}

// Vector CSR Kernel - 一个 Warp (32线程) 处理一行
__global__ void spmv_csr_vector_kernel(
    int num_rows,
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* x,
    cudaTextureObject_t tex_x,
    bool use_texture,
    float* y
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptrs[warp_id];
        int row_end = row_ptrs[warp_id + 1];

        // Warp 内线程协作处理一行
        for (int j = row_start + lane_id; j < row_end; j += 32) {
            sum += values[j] * fetch_x(x, tex_x, use_texture, col_indices[j]);
        }

        // Warp 级归约
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}

// Scalar CSR Kernel - 一个线程处理一行
__global__ void spmv_csr_scalar_kernel(
    int num_rows,
    const int* row_ptrs,
    const int* col_indices,
    const float* values,
    const float* x,
    cudaTextureObject_t tex_x,
    bool use_texture,
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * fetch_x(x, tex_x, use_texture, col_indices[j]);
        }
        y[row] = sum;
    }
}

// ELL Kernel
__global__ void spmv_ell_kernel(
    int num_rows,
    int max_nnz_per_row,
    const int* col_indices,
    const float* values,
    const float* x,
    cudaTextureObject_t tex_x,
    bool use_texture,
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_nnz_per_row; k++) {
            int idx = k * num_rows + row;
            int col = col_indices[idx];
            if (col >= 0) {
                sum += values[idx] * fetch_x(x, tex_x, use_texture, col);
            }
        }
        y[row] = sum;
    }
}

static bool is_valid_block_size(int block_size) {
    return block_size >= 32 && block_size <= 1024 && (block_size % 32 == 0);
}

static bool is_valid_csr_kernel_type(SpMVConfig::KernelType kernel_type) {
    switch (kernel_type) {
        case SpMVConfig::SCALAR_CSR:
        case SpMVConfig::VECTOR_CSR:
        case SpMVConfig::MERGE_PATH:
            return true;
        default:
            return false;
    }
}

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config, int vec_size,
                    SpMVExecutionContext* context) {
    SpMVResult result;

    if (!A || A->num_rows < 0 || A->num_cols < 0 || A->nnz < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int x_length = (vec_size >= 0) ? vec_size : A->num_cols;
    if (x_length < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    if (vec_size >= 0 && !spmv_validate_dimensions(A->num_cols, vec_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return result;
    }

    if ((x_length > 0 && !d_x) || (A->num_rows > 0 && !d_y)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    if (!A->d_row_ptrs || (A->nnz > 0 && (!A->d_col_indices || !A->d_values))) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    SpMVConfig default_config;
    if (!config) config = &default_config;

    if (!is_valid_csr_kernel_type(config->kernel_type) || !is_valid_block_size(config->block_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    result.y = d_y;

    if (A->num_rows == 0) {
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    if (A->nnz == 0 || x_length == 0) {
        cudaError_t err = cudaMemset(d_y, 0, A->num_rows * sizeof(float));
        result.error_code = (err == cudaSuccess)
            ? static_cast<int>(SpMVError::SUCCESS)
            : static_cast<int>(SpMVError::CUDA_MEMCPY);
        return result;
    }

    ScopedTexture fallback_texture;
    bool use_texture = config->use_texture;
    size_t texture_length = static_cast<size_t>(x_length);
    cudaTextureObject_t tex_x = 0;

    if (use_texture && texture_length > 0) {
        if (context) {
            int tex_status = prepare_texture_context(context, d_x, texture_length, use_texture, &tex_x,
                                                     &use_texture);
            if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
                result.error_code = tex_status;
                return result;
            }
        } else {
            int tex_status = fallback_texture.create(d_x, texture_length);
            if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
                result.error_code = tex_status;
                return result;
            }
            tex_x = fallback_texture.tex;
        }
    } else {
        use_texture = false;
        if (context) {
            context->reset();
        }
    }

    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;

    auto host_start = std::chrono::steady_clock::now();
    CudaTimer timer;
    bool use_cuda_timer = timer.init_status() == cudaSuccess;

    if (use_cuda_timer) {
        cudaError_t err = timer.record_start();
        if (err != cudaSuccess) {
            use_cuda_timer = false;
        }
    }

    cudaError_t err = cudaSuccess;
    switch (config->kernel_type) {
        case SpMVConfig::MERGE_PATH: {
            err = cudaMemsetAsync(d_y, 0, A->num_rows * sizeof(float));
            if (err != cudaSuccess) {
                result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
                return result;
            }
            spmv_csr_merge_path_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->nnz, A->d_row_ptrs, A->d_col_indices,
                A->d_values, d_x, tex_x, use_texture, d_y
            );
            break;
        }
        case SpMVConfig::VECTOR_CSR: {
            int warps_per_block = block_size / 32;
            int num_warps = (A->num_rows + warps_per_block - 1) / warps_per_block;
            spmv_csr_vector_kernel<<<num_warps, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices,
                A->d_values, d_x, tex_x, use_texture, d_y
            );
            break;
        }
        case SpMVConfig::SCALAR_CSR:
            spmv_csr_scalar_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices,
                A->d_values, d_x, tex_x, use_texture, d_y
            );
            break;
        default:
            result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
            return result;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        result.error_code = map_cuda_error(err, SpMVError::KERNEL_LAUNCH);
        return result;
    }

    float elapsed_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - host_start).count();
    if (use_cuda_timer) {
        err = timer.record_stop();
        if (err == cudaSuccess) {
            float timer_elapsed_ms = 0.0f;
            err = timer.elapsed_ms(&timer_elapsed_ms);
            if (err == cudaSuccess) {
                elapsed_ms = timer_elapsed_ms;
            }
        }
    }

    result.elapsed_ms = elapsed_ms;
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_csr(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
}

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config, int vec_size,
                    SpMVExecutionContext* context) {
    SpMVResult result;

    if (!A || A->num_rows < 0 || A->num_cols < 0 || A->max_nnz_per_row < 0 || A->nnz < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int x_length = (vec_size >= 0) ? vec_size : A->num_cols;
    if (x_length < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    if (vec_size >= 0 && !spmv_validate_dimensions(A->num_cols, vec_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return result;
    }

    if ((x_length > 0 && !d_x) || (A->num_rows > 0 && !d_y)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    size_t storage_size = static_cast<size_t>(A->num_rows) * static_cast<size_t>(A->max_nnz_per_row);
    if (storage_size > 0 && (!A->d_col_indices || !A->d_values)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    SpMVConfig default_config;
    default_config.kernel_type = SpMVConfig::ELL_KERNEL;
    if (!config) config = &default_config;

    if (config->kernel_type != SpMVConfig::ELL_KERNEL || !is_valid_block_size(config->block_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    result.y = d_y;

    if (A->num_rows == 0) {
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    if (storage_size == 0 || x_length == 0) {
        cudaError_t err = cudaMemset(d_y, 0, A->num_rows * sizeof(float));
        result.error_code = (err == cudaSuccess)
            ? static_cast<int>(SpMVError::SUCCESS)
            : static_cast<int>(SpMVError::CUDA_MEMCPY);
        return result;
    }

    ScopedTexture fallback_texture;
    bool use_texture = config->use_texture;
    size_t texture_length = static_cast<size_t>(x_length);
    cudaTextureObject_t tex_x = 0;

    if (use_texture && texture_length > 0) {
        if (context) {
            int tex_status = prepare_texture_context(context, d_x, texture_length, use_texture, &tex_x,
                                                     &use_texture);
            if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
                result.error_code = tex_status;
                return result;
            }
        } else {
            int tex_status = fallback_texture.create(d_x, texture_length);
            if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
                result.error_code = tex_status;
                return result;
            }
            tex_x = fallback_texture.tex;
        }
    } else {
        use_texture = false;
        if (context) {
            context->reset();
        }
    }

    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;

    auto host_start = std::chrono::steady_clock::now();
    CudaTimer timer;
    bool use_cuda_timer = timer.init_status() == cudaSuccess;
    if (use_cuda_timer) {
        cudaError_t timer_err = timer.record_start();
        if (timer_err != cudaSuccess) {
            use_cuda_timer = false;
        }
    }

    cudaError_t err = cudaSuccess;
    spmv_ell_kernel<<<num_blocks, block_size>>>(
        A->num_rows, A->max_nnz_per_row,
        A->d_col_indices, A->d_values, d_x, tex_x, use_texture, d_y
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        result.error_code = map_cuda_error(err, SpMVError::KERNEL_LAUNCH);
        return result;
    }

    float elapsed_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - host_start).count();
    if (use_cuda_timer) {
        err = timer.record_stop();
        if (err == cudaSuccess) {
            float timer_elapsed_ms = 0.0f;
            err = timer.elapsed_ms(&timer_elapsed_ms);
            if (err == cudaSuccess) {
                elapsed_ms = timer_elapsed_ms;
            }
        }
    }

    result.elapsed_ms = elapsed_ms;
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_ell(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
}

} // namespace spmv
