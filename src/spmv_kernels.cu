#include "spmv/bandwidth.h"
#include "spmv/spmv.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "internal/csr_device.h"
#include "internal/ell_device.h"

namespace cg = cooperative_groups;

namespace spmv {

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
        if (start)
            cudaEventDestroy(start);
        if (stop)
            cudaEventDestroy(stop);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    cudaError_t init_status() const { return status; }

    cudaError_t record_start(cudaStream_t stream) const {
        return (status == cudaSuccess) ? cudaEventRecord(start, stream) : status;
    }

    cudaError_t record_stop(cudaStream_t stream) const {
        if (status != cudaSuccess)
            return status;
        cudaError_t err = cudaEventRecord(stop, stream);
        if (err != cudaSuccess)
            return err;
        return cudaEventSynchronize(stop);
    }

    cudaError_t elapsed_ms(float* ms) const {
        if (!ms)
            return cudaErrorInvalidValue;
        if (status != cudaSuccess)
            return status;
        return cudaEventElapsedTime(ms, start, stop);
    }
};

static int map_cuda_error(cudaError_t err, SpMVError fallback) {
    if (err == cudaSuccess)
        return static_cast<int>(SpMVError::SUCCESS);
    if (err == cudaErrorMemoryAllocation)
        return static_cast<int>(SpMVError::CUDA_MALLOC);
    return static_cast<int>(fallback);
}

// ---------- Kernels ----------

__global__ void spmv_csr_scalar_kernel(int num_rows, const int* __restrict__ row_ptrs,
                                       const int* __restrict__ col_indices,
                                       const float* __restrict__ values,
                                       const float* __restrict__ x, float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * __ldg(&x[col_indices[j]]);
        }
        y[row] = sum;
    }
}

__global__ void spmv_csr_vector_kernel(int num_rows, const int* __restrict__ row_ptrs,
                                       const int* __restrict__ col_indices,
                                       const float* __restrict__ values,
                                       const float* __restrict__ x, float* __restrict__ y) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warp_id < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptrs[warp_id];
        int row_end = row_ptrs[warp_id + 1];

        for (int j = row_start + warp.thread_rank(); j < row_end; j += 32) {
            sum += values[j] * __ldg(&x[col_indices[j]]);
        }

#if __CUDA_ARCH__ >= 800
        float total = cg::reduce(warp, sum, cg::plus<float>());
#else
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        float total = sum;
#endif

        if (warp.thread_rank() == 0) {
            y[warp_id] = total;
        }
    }
}

__device__ int merge_path_find_row(const int* __restrict__ row_ptrs, int num_rows, int nz_index) {
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

__global__ void spmv_csr_merge_path_kernel(int num_rows, int nnz, const int* __restrict__ row_ptrs,
                                           const int* __restrict__ col_indices,
                                           const float* __restrict__ values,
                                           const float* __restrict__ x, float* __restrict__ y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    if (nnz <= 0)
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
        sum += values[nz] * __ldg(&x[col_indices[nz]]);
    }

    if (current_row < num_rows) {
        atomicAdd(&y[current_row], sum);
    }
}

__global__ void spmv_ell_kernel(int num_rows, int max_nnz_per_row,
                                const int* __restrict__ col_indices,
                                const float* __restrict__ values, const float* __restrict__ x,
                                float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_nnz_per_row; k++) {
            // 64-bit index math: num_rows * max_nnz_per_row can exceed
            // INT_MAX for large ELL matrices (host side uses size_t).
            long long idx = static_cast<long long>(k) * num_rows + row;
            int col = col_indices[idx];
            if (col >= 0) {
                sum += values[idx] * __ldg(&x[col]);
            }
        }
        y[row] = sum;
    }
}

// ---------- Helpers ----------

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

// Applies an L2 persisting access-policy window for d_x while in scope and
// restores the stream's previous window on destruction, so the hint never
// leaks into unrelated work the caller later enqueues on the same stream.
struct L2PersistingScope {
    cudaStream_t stream = nullptr;
    cudaStreamAttrValue saved{};
    bool applied = false;

    explicit L2PersistingScope(cudaStream_t s) : stream(s) {}

    ~L2PersistingScope() {
        if (applied) {
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &saved);
        }
    }

    L2PersistingScope(const L2PersistingScope&) = delete;
    L2PersistingScope& operator=(const L2PersistingScope&) = delete;

    void apply(const float* d_x, size_t x_length) {
        if (!d_x || x_length == 0)
            return;

        // Best effort: if the previous window cannot be read, restore a
        // zeroed window later (num_bytes == 0 disables the policy).
        if (cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &saved) !=
            cudaSuccess) {
            saved = cudaStreamAttrValue{};
        }

        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr = const_cast<float*>(d_x);
        attr.accessPolicyWindow.num_bytes = x_length * sizeof(float);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) ==
            cudaSuccess) {
            applied = true;
        }
    }
};

static int synchronize_and_check(cudaStream_t stream) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return static_cast<int>(SpMVError::KERNEL_LAUNCH);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return map_cuda_error(err, SpMVError::KERNEL_LAUNCH);
    return static_cast<int>(SpMVError::SUCCESS);
}

static float resolve_elapsed_ms(const CudaTimer& timer, bool use_cuda_timer, float fallback_ms) {
    if (!use_cuda_timer)
        return fallback_ms;
    float ms = 0.0f;
    return (timer.elapsed_ms(&ms) == cudaSuccess) ? ms : fallback_ms;
}

static void finalize_result(SpMVResult& result, float elapsed_ms, int nnz, float bandwidth_gb_s) {
    result.elapsed_ms = elapsed_ms;
    result.gflops = (elapsed_ms > 0.0f) ? (2.0f * nnz) / (elapsed_ms * 1e6f) : 0.0f;
    result.bandwidth_gb_s = bandwidth_gb_s;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
}

// ---------- Public API ----------

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size, cudaStream_t stream) {
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
        cudaError_t err = cudaMemsetAsync(d_y, 0, A->num_rows * sizeof(float), stream);
        if (err != cudaSuccess) {
            result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
            return result;
        }
        if (config->enable_timing) {
            // Keep blocking semantics consistent with the kernel path: when
            // timing is enabled, d_y is complete once the call returns.
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
                return result;
            }
        }
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    L2PersistingScope l2_scope(stream);
    l2_scope.apply(d_x, static_cast<size_t>(x_length));

    int block_size = config->block_size;

    auto host_start = std::chrono::steady_clock::now();
    CudaTimer timer;
    bool use_cuda_timer = false;
    if (config->enable_timing) {
        use_cuda_timer = (timer.init_status() == cudaSuccess);
        if (use_cuda_timer && timer.record_start(stream) != cudaSuccess) {
            use_cuda_timer = false;
        }
    }

    switch (config->kernel_type) {
        case SpMVConfig::MERGE_PATH: {
            cudaError_t err = cudaMemsetAsync(d_y, 0, A->num_rows * sizeof(float), stream);
            if (err != cudaSuccess) {
                result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
                return result;
            }
            // Partition the grid by nnz (merge path's unit of work), not by
            // num_rows: the skewed matrices routed to this kernel often have
            // few rows but very long ones, and a rows-based grid starves the
            // GPU in exactly the case merge path exists for.
            constexpr int kWorkPerThread = 32;
            // (n - 1) / d + 1 computes ceil(n / d) without overflowing int.
            int total_threads = (A->nnz - 1) / kWorkPerThread + 1;
            int merge_blocks = (total_threads - 1) / block_size + 1;
            spmv_csr_merge_path_kernel<<<merge_blocks, block_size, 0, stream>>>(
                A->num_rows, A->nnz, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A), d_x,
                d_y);
            break;
        }
        case SpMVConfig::VECTOR_CSR: {
            int warps_per_block = block_size / 32;
            int num_warps = (A->num_rows - 1) / warps_per_block + 1;
            spmv_csr_vector_kernel<<<num_warps, block_size, 0, stream>>>(
                A->num_rows, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A), d_x, d_y);
            break;
        }
        case SpMVConfig::SCALAR_CSR: {
            int num_blocks = (A->num_rows - 1) / block_size + 1;
            spmv_csr_scalar_kernel<<<num_blocks, block_size, 0, stream>>>(
                A->num_rows, csr_d_row_ptrs(A), csr_d_col_indices(A), csr_d_values(A), d_x, d_y);
            break;
        }
        default:
            result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
            return result;
    }

    if (!config->enable_timing) {
        // Async mode: the kernel stays in flight on `stream`.  Only
        // synchronous launch failures are reported here; the caller is
        // responsible for synchronizing before reading d_y and for
        // observing asynchronous errors.
        cudaError_t err = cudaGetLastError();
        result.error_code = (err == cudaSuccess) ? static_cast<int>(SpMVError::SUCCESS)
                                                 : static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }

    int sync_status = synchronize_and_check(stream);
    if (sync_status != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = sync_status;
        return result;
    }

    if (use_cuda_timer)
        timer.record_stop(stream);
    float fallback_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - host_start)
            .count();
    float elapsed_ms = resolve_elapsed_ms(timer, use_cuda_timer, fallback_ms);
    // Derive bandwidth from the reported elapsed_ms so the fields of
    // SpMVResult are mutually consistent.
    float bw = compute_bandwidth_csr(A, elapsed_ms).achieved_bandwidth_gb_s;
    finalize_result(result, elapsed_ms, A->nnz, bw);

    return result;
}

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y, const SpMVConfig* config,
                    int vec_size, cudaStream_t stream) {
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
        cudaError_t err = cudaMemsetAsync(d_y, 0, A->num_rows * sizeof(float), stream);
        if (err != cudaSuccess) {
            result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
            return result;
        }
        if (config->enable_timing) {
            // Keep blocking semantics consistent with the kernel path: when
            // timing is enabled, d_y is complete once the call returns.
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                result.error_code = map_cuda_error(err, SpMVError::CUDA_MEMCPY);
                return result;
            }
        }
        result.error_code = static_cast<int>(SpMVError::SUCCESS);
        return result;
    }

    L2PersistingScope l2_scope(stream);
    l2_scope.apply(d_x, static_cast<size_t>(x_length));

    int block_size = config->block_size;
    int num_blocks = (A->num_rows - 1) / block_size + 1;

    auto host_start = std::chrono::steady_clock::now();
    CudaTimer timer;
    bool use_cuda_timer = false;
    if (config->enable_timing) {
        use_cuda_timer = (timer.init_status() == cudaSuccess);
        if (use_cuda_timer && timer.record_start(stream) != cudaSuccess) {
            use_cuda_timer = false;
        }
    }

    spmv_ell_kernel<<<num_blocks, block_size, 0, stream>>>(
        A->num_rows, A->max_nnz_per_row, ell_d_col_indices(A), ell_d_values(A), d_x, d_y);

    if (!config->enable_timing) {
        // Async mode: see spmv_csr for the contract.
        cudaError_t err = cudaGetLastError();
        result.error_code = (err == cudaSuccess) ? static_cast<int>(SpMVError::SUCCESS)
                                                 : static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }

    int sync_status = synchronize_and_check(stream);
    if (sync_status != static_cast<int>(SpMVError::SUCCESS)) {
        result.error_code = sync_status;
        return result;
    }

    if (use_cuda_timer)
        timer.record_stop(stream);
    float fallback_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - host_start)
            .count();
    float elapsed_ms = resolve_elapsed_ms(timer, use_cuda_timer, fallback_ms);
    // Derive bandwidth from the reported elapsed_ms so the fields of
    // SpMVResult are mutually consistent.
    float bw = compute_bandwidth_ell(A, elapsed_ms).achieved_bandwidth_gb_s;
    finalize_result(result, elapsed_ms, A->nnz, bw);

    return result;
}

}  // namespace spmv
