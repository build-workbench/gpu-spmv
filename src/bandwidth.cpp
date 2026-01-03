#include "spmv/bandwidth.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace spmv {

float get_gpu_peak_bandwidth() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 理论峰值带宽 = 内存时钟频率 * 内存总线宽度 * 2 (DDR)
    // 单位: GB/s
    float memory_clock_khz = prop.memoryClockRate;  // kHz
    float memory_bus_width = prop.memoryBusWidth;   // bits
    
    // 转换为 GB/s
    float bandwidth_gb_s = (memory_clock_khz * 1000.0f) * (memory_bus_width / 8.0f) * 2.0f / 1e9f;
    
    return bandwidth_gb_s;
}

BandwidthMetrics compute_bandwidth_csr(const CSRMatrix* A, float elapsed_ms) {
    BandwidthMetrics metrics;
    
    if (!A || elapsed_ms <= 0.0f) {
        return metrics;
    }
    
    // 计算传输的字节数
    // 读取: values (nnz * sizeof(float)) + col_indices (nnz * sizeof(int)) 
    //      + row_ptrs ((num_rows + 1) * sizeof(int)) + x (num_cols * sizeof(float))
    // 写入: y (num_rows * sizeof(float))
    
    size_t bytes_read = 0;
    bytes_read += A->nnz * sizeof(float);           // values
    bytes_read += A->nnz * sizeof(int);             // col_indices
    bytes_read += (A->num_rows + 1) * sizeof(int);  // row_ptrs
    bytes_read += A->num_cols * sizeof(float);      // x vector
    
    size_t bytes_write = A->num_rows * sizeof(float);  // y vector
    
    size_t total_bytes = bytes_read + bytes_write;
    
    // 计算带宽 (GB/s)
    float elapsed_s = elapsed_ms / 1000.0f;
    metrics.achieved_bandwidth_gb_s = (total_bytes / 1e9f) / elapsed_s;
    
    metrics.theoretical_bandwidth_gb_s = get_gpu_peak_bandwidth();
    
    if (metrics.theoretical_bandwidth_gb_s > 0.0f) {
        metrics.efficiency = metrics.achieved_bandwidth_gb_s / metrics.theoretical_bandwidth_gb_s;
        metrics.efficiency = std::min(metrics.efficiency, 1.0f);  // Cap at 100%
    }
    
    return metrics;
}

BandwidthMetrics compute_bandwidth_ell(const ELLMatrix* A, float elapsed_ms) {
    BandwidthMetrics metrics;
    
    if (!A || elapsed_ms <= 0.0f) {
        return metrics;
    }
    
    // ELL 格式的内存访问
    size_t ell_size = static_cast<size_t>(A->num_rows) * A->max_nnz_per_row;
    
    size_t bytes_read = 0;
    bytes_read += ell_size * sizeof(float);     // values
    bytes_read += ell_size * sizeof(int);       // col_indices
    bytes_read += A->num_cols * sizeof(float);  // x vector
    
    size_t bytes_write = A->num_rows * sizeof(float);  // y vector
    
    size_t total_bytes = bytes_read + bytes_write;
    
    float elapsed_s = elapsed_ms / 1000.0f;
    metrics.achieved_bandwidth_gb_s = (total_bytes / 1e9f) / elapsed_s;
    
    metrics.theoretical_bandwidth_gb_s = get_gpu_peak_bandwidth();
    
    if (metrics.theoretical_bandwidth_gb_s > 0.0f) {
        metrics.efficiency = metrics.achieved_bandwidth_gb_s / metrics.theoretical_bandwidth_gb_s;
        metrics.efficiency = std::min(metrics.efficiency, 1.0f);
    }
    
    return metrics;
}

} // namespace spmv
