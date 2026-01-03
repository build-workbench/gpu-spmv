#include "spmv/spmv.h"
#include "spmv/bandwidth.h"
#include <cuda_runtime.h>

namespace spmv {

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
                sum += values[current_nz] * x[col_indices[current_nz]];
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
            sum += values[j] * x[col_indices[j]];
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
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
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
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_nnz_per_row; k++) {
            int idx = k * num_rows + row;  // Column-major
            int col = col_indices[idx];
            if (col >= 0) {
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}

SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, 
                    const SpMVConfig* config) {
    SpMVResult result;
    
    if (!A || !d_x || !d_y) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }
    
    if (!A->d_row_ptrs || !A->d_col_indices || (A->nnz > 0 && !A->d_values)) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }
    
    SpMVConfig default_config;
    if (!config) {
        config = &default_config;
    }
    
    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    switch (config->kernel_type) {
        case SpMVConfig::MERGE_PATH: {
            // 初始化输出为零
            cudaMemset(d_y, 0, A->num_rows * sizeof(float));
            
            int total_work = A->num_rows + A->nnz;
            int num_threads = block_size * num_blocks;
            spmv_csr_merge_path_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->nnz, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, d_y
            );
            break;
        }
        case SpMVConfig::VECTOR_CSR: {
            // Vector CSR: 一个 Warp 处理一行
            int warps_per_block = block_size / 32;
            int num_warps = (A->num_rows + warps_per_block - 1) / warps_per_block;
            spmv_csr_vector_kernel<<<num_warps, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, d_y
            );
            break;
        }
        case SpMVConfig::SCALAR_CSR:
        default:
            spmv_csr_scalar_kernel<<<num_blocks, block_size>>>(
                A->num_rows, A->d_row_ptrs, A->d_col_indices, 
                A->d_values, d_x, d_y
            );
            break;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return result;
    }
    
    cudaEventElapsedTime(&result.elapsed_ms, start, stop);
    
    // 计算 GFLOPS: 2 * nnz operations (multiply + add)
    result.gflops = (2.0f * A->nnz) / (result.elapsed_ms * 1e6f);
    
    // 计算带宽度量
    BandwidthMetrics bw = compute_bandwidth_csr(A, result.elapsed_ms);
    result.bandwidth_gb_s = bw.achieved_bandwidth_gb_s;
    
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config) {
    SpMVResult result;
    
    if (!A || !d_x || !d_y) {
        result.error_code = static_cast<int>(SpMVError::INVALID_ARGUMENT);
        return result;
    }
    
    if (!A->d_col_indices || !A->d_values) {
        result.error_code = static_cast<int>(SpMVError::INVALID_FORMAT);
        return result;
    }
    
    SpMVConfig default_config;
    if (!config) {
        config = &default_config;
    }
    
    int block_size = config->block_size;
    int num_blocks = (A->num_rows + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    spmv_ell_kernel<<<num_blocks, block_size>>>(
        A->num_rows, A->max_nnz_per_row,
        A->d_col_indices, A->d_values, d_x, d_y
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_code = static_cast<int>(SpMVError::KERNEL_LAUNCH);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return result;
    }
    
    cudaEventElapsedTime(&result.elapsed_ms, start, stop);
    
    // 计算实际非零元素数
    int actual_nnz = 0;
    for (int i = 0; i < A->num_rows; i++) {
        for (int k = 0; k < A->max_nnz_per_row; k++) {
            int idx = k * A->num_rows + i;
            if (A->col_indices[idx] >= 0) actual_nnz++;
        }
    }
    
    result.gflops = (2.0f * actual_nnz) / (result.elapsed_ms * 1e6f);
    
    // 计算带宽度量
    BandwidthMetrics bw = compute_bandwidth_ell(A, result.elapsed_ms);
    result.bandwidth_gb_s = bw.achieved_bandwidth_gb_s;
    
    result.y = d_y;
    result.error_code = static_cast<int>(SpMVError::SUCCESS);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}

} // namespace spmv
