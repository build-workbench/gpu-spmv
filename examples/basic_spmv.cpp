// Minimal end-to-end example: build a sparse matrix, run SpMV, print y.
//
// With CUDA (default build) the product runs on the GPU via spmv_csr.
// In CPU-only builds (-DSPMV_REQUIRE_CUDA=OFF) it falls back to the CPU
// reference path so the example still runs.

#include <cstdio>
#include <spmv/csr_matrix.h>
#include <spmv/spmv.h>
#include <vector>

#if SPMV_WITH_CUDA
#include <spmv/cuda_buffer.h>
#endif

int main() {
    const float dense[] = {
        1.0f, 0.0f, 2.0f,  //
        0.0f, 3.0f, 4.0f,  //
        0.0f, 0.0f, 5.0f,  //
    };
    const float h_x[] = {1.0f, 1.0f, 1.0f};
    const int n = 3;

    spmv::CSRMatrix* csr = spmv::csr_create(n, n, 0);
    if (!csr) {
        std::fprintf(stderr, "csr_create failed\n");
        return 1;
    }
    if (spmv::csr_from_dense(csr, dense, n, n) != 0) {
        std::fprintf(stderr, "csr_from_dense failed\n");
        spmv::csr_destroy(csr);
        return 1;
    }

    std::vector<float> y(n, 0.0f);

#if SPMV_WITH_CUDA
    if (spmv::csr_to_gpu(csr) != 0) {
        std::fprintf(stderr, "csr_to_gpu failed\n");
        spmv::csr_destroy(csr);
        return 1;
    }

    spmv::CudaBuffer<float> d_x(n);
    spmv::CudaBuffer<float> d_y(n);
    d_x.copyFromHost(h_x, n);

    spmv::SpMVConfig config = spmv::spmv_auto_config(csr);
    spmv::SpMVResult result = spmv::spmv_csr(csr, d_x.get(), d_y.get(), &config, n);
    if (spmv::spmv_result_error(result) != spmv::SpMVError::SUCCESS) {
        std::fprintf(stderr, "spmv_csr failed: %s\n",
                     spmv::spmv_error_string(spmv::spmv_result_error(result)));
        spmv::csr_destroy(csr);
        return 1;
    }

    d_y.copyToHost(y.data(), n);
    std::printf("GPU SpMV: y = [%g, %g, %g]  (%.1f GFLOP/s, %.1f GB/s)\n", y[0], y[1], y[2],
                result.gflops, result.bandwidth_gb_s);
#else
    if (spmv::spmv_cpu_csr(csr, h_x, y.data()) != 0) {
        std::fprintf(stderr, "spmv_cpu_csr failed\n");
        spmv::csr_destroy(csr);
        return 1;
    }
    std::printf("CPU SpMV: y = [%g, %g, %g]\n", y[0], y[1], y[2]);
#endif

    spmv::csr_destroy(csr);
    return 0;
}
