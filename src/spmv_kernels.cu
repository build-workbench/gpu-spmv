#include "spmv/spmv.h"
#include "spmv/bandwidth.h"
#include <cuda_runtime.h>

namespace spmv {

// ---------- RAII helpers ----------

// RAII wrapper for CUDA events (timing)
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void record_start() { cudaEventRecord(start); }
    void record_stop()  { cudaEventRecord(stop); cudaEventSynchronize(stop); }
    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// RAII wrapper for a CUDA texture object
struct ScopedTexture {
    cudaTextureObject_t tex = 0;
    bool valid = false;

    ScopedTexture() = default;
    ~ScopedTexture() { if (valid) cudaDestroyTextureObject(tex); }
    ScopedTexture(const ScopedTexture&) = delete;
    ScopedTexture& operator=(const ScopedTexture&) = delete;

    // Returns SpMVError code
    int create(const float* d_x, size_t count) {
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
                // 完成当前行
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
            int idx = k * num_rows + row;  // Column-major
            int col = col_indices[idx];
            if (col >= 0) {
                sum += values[idx] * fetch_x(x, tex_x, use_texture, col);
            }
        }
        y[row] = sum;
    }
}

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, 
                    const SpMVConfig* config, int vec_size) {
    SpMVResult result;
    
    if (!A || !d_x || !d_y) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    if (vec_size >= 0 && !spmv_validate_dimensions(A->num_cols, vec_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return result;
    }
    
    if (!A->d_row_ptrs || !A->d_col_indices || (A->nnz > 0 && !A->d_values)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }
    
    SpMVConfig default_config;
    if (!config) config = &default_config;

    // 纹理对象 (RAII)
    ScopedTexture stex;
    bool use_texture = config->use_texture;
    size_t x_length = vec_size >= 0 ? static_cast<size_t>(vec_size)
                                    : static_cast<size_t>(A->num_cols);
    if (use_texture && x_length > 0) {
        int tex_status = stex.create(d_x, x_length);
        if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
            result.error_code = tex_status;
            return result;
        }
    } else {
        use_texture = false;
    }
    
    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;
    
    CudaTimer timer;
    timer.record_start();
    
    switch (config->kernel_type) {
        case SpMVConfig::MERGE_PATH: {
            cudaMemsetAsync(d_y, 0, A->num_rows * sizeof(float));
            spmv_csr_merge_path_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->nnz, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, stex.tex, use_texture, d_y
            );
            break;
        }
        case SpMVConfig::VECTOR_CSR: {
            int warps_per_block = block_size / 32;
            int num_warps = (A->num_rows + warps_per_block - 1) / warps_per_block;
            spmv_csr_vector_kernel<<<num_warps, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, stex.tex, use_texture, d_y
            );
            break;
        }
        case SpMVConfig::SCALAR_CSR:
        default:
            spmv_csr_scalar_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, stex.tex, use_texture, d_y
            );
            break;
    }
    
    timer.record_stop();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }
    
    result.elapsed_ms = timer.elapsed_ms();
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_csr(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
    
    return result;
}

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config, int vec_size) {
    SpMVResult result;
    
    if (!A || !d_x || !d_y) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }

    if (vec_size >= 0 && !spmv_validate_dimensions(A->num_cols, vec_size)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_DIMENSION);
        return result;
    }
    
    if (!A->d_col_indices || !A->d_values) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }
    
    SpMVConfig default_config;
    if (!config) config = &default_config;

    // 纹理对象 (RAII)
    ScopedTexture stex;
    bool use_texture = config->use_texture;
    size_t x_length = vec_size >= 0 ? static_cast<size_t>(vec_size)
                                    : static_cast<size_t>(A->num_cols);
    if (use_texture && x_length > 0) {
        int tex_status = stex.create(d_x, x_length);
        if (tex_status != static_cast<int>(SpMVError::SUCCESS)) {
            result.error_code = tex_status;
            return result;
        }
    } else {
        use_texture = false;
    }
    
    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;
    
    CudaTimer timer;
    timer.record_start();
    
    spmv_ell_kernel<<<num_blocks, block_size>>>(
        A->num_rows, A->max_nnz_per_row,
        A->d_col_indices, A->d_values, d_x, stex.tex, use_texture, d_y
    );
    
    timer.record_stop();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        return result;
    }
    
    result.elapsed_ms = timer.elapsed_ms();
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    result.bandwidth_gb_s = compute_bandwidth_ell(A, result.elapsed_ms).achieved_bandwidth_gb_s;
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
    
    return result;
}

} // namespace spmv
