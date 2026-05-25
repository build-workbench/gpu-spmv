#include "internal/csr_device.h"
#include "internal/ell_device.h"
#include "internal/texture_cache.h"
#include "spmv/bandwidth.h"
#include "spmv/spmv.h"

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

    cudaError_t record_start() const {
        return (status == cudaSuccess) ? cudaEventRecord(start) : status;
    }

    cudaError_t record_stop() const {
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
        if (!d_x || count == 0)
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);

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
        if (err != cudaSuccess)
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        valid = true;
        return static_cast<int>(SpMVError::SUCCESS);
    }
};

__device__ __forceinline__ float fetch_x(const float* x, cudaTextureObject_t tex_x,
                                         bool use_texture, int idx) {
    return use_texture ? tex1Dfetch<float>(tex_x, idx) : x[idx];
}

__device__ int merge_path_find_row(const int* row_ptrs, int num_rows, int nz_index) {
    int low = 0;
    int high = num_rows - 1;

    while (low < high) {
        int mid = low + (high - low) / 2;
        if (row_ptrs[mid + 1] <= nz_index) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low;
}

// Merge Path Kernel
__global__ void spmv_csr_merge_path_kernel(int num_rows, int nnz, const int* row_ptrs,
                                           const int* col_indices, const float* values,
                                           const float* x, cudaTextureObject_t tex_x,
                                           bool use_texture, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (tid >= total_threads || nnz <= 0)
        return;

    int nz_start = static_cast<int>((static_cast<long long>(tid) * nnz) / total_threads);
    int nz_end = static_cast<int>((static_cast<long long>(tid + 1) * nnz) / total_threads);

    if (nz_start >= nz_end)
        return;

    int current_row = merge_path_find_row(row_ptrs, num_rows, nz_start);
    float sum = 0.0f;

    for (int nz = nz_start; nz < nz_end; ++nz) {
        while (current_row + 1 < num_rows && row_ptrs[current_row + 1] <= nz) {
            atomicAdd(&y[current_row], sum);
            sum = 0.0f;
            current_row++;
        }

        sum += values[nz] * fetch_x(x, tex_x, use_texture, col_indices[nz]);
    }

    if (current_row < num_rows) {
        atomicAdd(&y[current_row], sum);
    }
}

// Vector CSR Kernel - 一个 Warp (32线程) 处理一行
__global__ void spmv_csr_vector_kernel(int num_rows, const int* row_ptrs, const int* col_indices,
                                       const float* values, const float* x,
                                       cudaTextureObject_t tex_x, bool use_texture, float* y) {
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
__global__ void spmv_csr_scalar_kernel(int num_rows, const int* row_ptrs, const int* col_indices,
                                       const float* values, const float* x,
                                       cudaTextureObject_t tex_x, bool use_texture, float* y) {
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
__global__ void spmv_ell_kernel(int num_rows, int max_nnz_per_row, const int* col_indices,
                                const float* values, const float* x, cudaTextureObject_t tex_x,
                                bool use_texture, float* y) {
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
    return block_size >= MIN_BLOCK_SIZE && block_size <= MAX_BLOCK_SIZE &&
           (block_size % WARP_SIZE == 0);
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

// ---------- Launch helpers (deepened module) ----------

// Validates vector arguments shared by spmv_csr and spmv_ell.
static bool validate_spmv_vectors(int num_cols, int num_rows, const float* d_x, float* d_y,
                                  int vec_size, int* out_x_length, SpMVResult* out_result) {
    int x_length = (vec_size >= 0) ? vec_size : num_cols;
    if (x_length < 0) {
        out_result->error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return false;
    }
    if (vec_size >= 0 && !spmv_validate_dimensions(num_cols, vec_size)) {
        out_result->error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return false;
    }
    if ((x_length > 0 && !d_x) || (num_rows > 0 && !d_y)) {
        out_result->error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return false;
    }
    *out_x_length = x_length;
    return true;
}

static int synchronize_and_check() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(SpMVError::KERNEL_LAUNCH);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return map_cuda_error(err, SpMVError::KERNEL_LAUNCH);
    }
    return static_cast<int>(SpMVError::SUCCESS);
}

static float read_timer_ms(const CudaTimer& timer, bool use_cuda_timer, float fallback_ms) {
    if (!use_cuda_timer)
        return fallback_ms;
    cudaError_t err = timer.record_stop();
    if (err != cudaSuccess)
        return fallback_ms;
    float ms = 0.0f;
    err = timer.elapsed_ms(&ms);
    return (err == cudaSuccess) ? ms : fallback_ms;
}

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* context) {
    SpMVResult result;

    if (!A || A->num_rows < 0 || A->num_cols < 0 || A->nnz < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int x_length = 0;
    if (!validate_spmv_vectors(A->num_cols, A->num_rows, d_x, d_y, vec_size, &x_length, &result)) {
        return result;
    }

    if (!csr_d_row_ptrs(A) || (A->nnz > 0 && (!csr_d_col_indices(A) || !csr_d_values(A)))) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    SpMVConfig default_config;
    if (!config)
        config = &default_config;

    if (!is_valid_csr_kernel_type(config->kernel_type) ||
        !is_valid_block_size(config->block_size)) {
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
        result.error_code = (err == cudaSuccess) ? static_cast<int>(SpMVError::SUCCESS)
                                                 : static_cast<int>(SpMVError::CUDA_MEMCPY);
        return result;
    }

    ScopedTexture fallback_texture;
    bool use_texture = config->use_texture;
    size_t texture_length = static_cast<size_t>(x_length);
    cudaTextureObject_t tex_x = 0;

    if (use_texture && texture_length > 0) {
        if (context) {
            int tex_status = spmv_prepare_texture(context, d_x, texture_length, use_texture, &tex_x,
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
                A->num_rows, A->nnz, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A),
                d_x, tex_x, use_texture, d_y);
            break;
        }
        case SpMVConfig::VECTOR_CSR: {
            int warps_per_block = block_size / 32;
            int num_warps = (A->num_rows + warps_per_block - 1) / warps_per_block;
            spmv_csr_vector_kernel<<<num_warps, block_size>>>(
                A->num_rows, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A), d_x, tex_x,
                use_texture, d_y);
            break;
        }
        case SpMVConfig::SCALAR_CSR:
            spmv_csr_scalar_kernel<<<num_blocks, block_size>>>(
                A->num_rows, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A), d_x, tex_x,
                use_texture, d_y);
            break;
        default:
            result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
            return result;
    }

    int sync_status = synchronize_and_check();
    if (sync_status != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = sync_status;
        return result;
    }

    float fallback_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - host_start)
            .count();
    result.elapsed_ms = read_timer_ms(timer, use_cuda_timer, fallback_ms);
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_csr(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
}

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size, SpMVExecutionContext* context) {
    SpMVResult result;

    if (!A || A->num_rows < 0 || A->num_cols < 0 || A->max_nnz_per_row < 0 || A->nnz < 0) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    int x_length = 0;
    if (!validate_spmv_vectors(A->num_cols, A->num_rows, d_x, d_y, vec_size, &x_length, &result)) {
        return result;
    }

    size_t storage_size =
        static_cast<size_t>(A->num_rows) * static_cast<size_t>(A->max_nnz_per_row);
    if (storage_size > 0 && (!ell_d_col_indices(A) || !ell_d_values(A))) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }

    SpMVConfig default_config;
    default_config.kernel_type = SpMVConfig::ELL_KERNEL;
    if (!config)
        config = &default_config;

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
        result.error_code = (err == cudaSuccess) ? static_cast<int>(SpMVError::SUCCESS)
                                                 : static_cast<int>(SpMVError::CUDA_MEMCPY);
        return result;
    }

    ScopedTexture fallback_texture;
    bool use_texture = config->use_texture;
    size_t texture_length = static_cast<size_t>(x_length);
    cudaTextureObject_t tex_x = 0;

    if (use_texture && texture_length > 0) {
        if (context) {
            int tex_status = spmv_prepare_texture(context, d_x, texture_length, use_texture, &tex_x,
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
    spmv_ell_kernel<<<num_blocks, block_size>>>(A->num_rows, A->max_nnz_per_row,
                                                ell_d_col_indices(A), ell_d_values(A), d_x, tex_x,
                                                use_texture, d_y);

    int sync_status = synchronize_and_check();
    if (sync_status != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = sync_status;
        return result;
    }

    float fallback_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - host_start)
            .count();
    result.elapsed_ms = read_timer_ms(timer, use_cuda_timer, fallback_ms);
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_ell(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);

    return result;
}

}  // namespace spmv
