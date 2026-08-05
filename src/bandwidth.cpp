#include "spmv/bandwidth.h"

#include "spmv/cuda_compat.h"

#include <algorithm>
#include <mutex>

namespace spmv {

float get_gpu_peak_bandwidth() {
    static float cached_bandwidth = -1.0f;
    static std::once_flag flag;

    std::call_once(flag, []() {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
            cached_bandwidth = 0.0f;
            return;
        }

        float memory_clock_khz = static_cast<float>(prop.memoryClockRate);
        float memory_bus_width = static_cast<float>(prop.memoryBusWidth);
        cached_bandwidth = (memory_clock_khz * 1000.0f) * (memory_bus_width / 8.0f) * 2.0f / 1e9f;
    });

    return cached_bandwidth;
}

static BandwidthMetrics make_metrics(size_t total_bytes, float elapsed_ms) {
    BandwidthMetrics metrics;
    if (elapsed_ms <= 0.0f)
        return metrics;

    float elapsed_s = elapsed_ms / 1000.0f;
    metrics.achieved_bandwidth_gb_s = (total_bytes / 1e9f) / elapsed_s;
    metrics.theoretical_bandwidth_gb_s = get_gpu_peak_bandwidth();

    if (metrics.theoretical_bandwidth_gb_s > 0.0f) {
        metrics.efficiency =
            std::min(metrics.achieved_bandwidth_gb_s / metrics.theoretical_bandwidth_gb_s, 1.0f);
    }
    return metrics;
}

BandwidthMetrics compute_bandwidth_csr(const CSRMatrix* A, float elapsed_ms) {
    if (!A || elapsed_ms <= 0.0f)
        return {};

    size_t bytes = 0;
    bytes += static_cast<size_t>(A->nnz) * sizeof(float);
    bytes += static_cast<size_t>(A->nnz) * sizeof(int);
    bytes += static_cast<size_t>(A->num_rows + 1) * sizeof(int);
    bytes += static_cast<size_t>(A->num_cols) * sizeof(float);
    bytes += static_cast<size_t>(A->num_rows) * sizeof(float);

    return make_metrics(bytes, elapsed_ms);
}

BandwidthMetrics compute_bandwidth_ell(const ELLMatrix* A, float elapsed_ms) {
    if (!A || elapsed_ms <= 0.0f)
        return {};

    size_t ell_size = static_cast<size_t>(A->num_rows) * A->max_nnz_per_row;
    size_t bytes = 0;
    bytes += ell_size * sizeof(float);
    bytes += ell_size * sizeof(int);
    bytes += static_cast<size_t>(A->num_cols) * sizeof(float);
    bytes += static_cast<size_t>(A->num_rows) * sizeof(float);

    return make_metrics(bytes, elapsed_ms);
}

}  // namespace spmv
