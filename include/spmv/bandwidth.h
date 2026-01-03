#ifndef SPMV_BANDWIDTH_H
#define SPMV_BANDWIDTH_H

#include "csr_matrix.h"
#include "ell_matrix.h"

namespace spmv {

// 带宽度量
struct BandwidthMetrics {
    float theoretical_bandwidth_gb_s;  // GPU 理论峰值带宽
    float achieved_bandwidth_gb_s;     // 实际达到的带宽
    float efficiency;                  // 带宽利用率 = achieved / theoretical
    
    BandwidthMetrics() : theoretical_bandwidth_gb_s(0.0f), 
                        achieved_bandwidth_gb_s(0.0f), 
                        efficiency(0.0f) {}
};

// 计算 CSR SpMV 的带宽
BandwidthMetrics compute_bandwidth_csr(const CSRMatrix* A, float elapsed_ms);

// 计算 ELL SpMV 的带宽
BandwidthMetrics compute_bandwidth_ell(const ELLMatrix* A, float elapsed_ms);

// 获取 GPU 理论峰值带宽
float get_gpu_peak_bandwidth();

} // namespace spmv

#endif // SPMV_BANDWIDTH_H
