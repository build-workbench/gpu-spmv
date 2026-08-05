// Synthetic SpMV benchmark.
//
// Generates CSR matrices with controllable row-length skew, runs every CSR
// kernel repeatedly, and reports median timing, GFLOP/s, and bandwidth.
// Optionally loads a real matrix when given a Matrix Market file:
//
//   spmv_bench                     # built-in synthetic cases
//   spmv_bench path/to/matrix.mtx  # benchmark a real matrix

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <spmv/csr_matrix.h>
#include <spmv/cuda_buffer.h>
#include <spmv/market_io.h>
#include <spmv/spmv.h>
#include <string>
#include <vector>

using namespace spmv;

namespace {

CSRMatrix* make_synthetic_csr(int rows, int cols, int min_nnz, int max_nnz, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nnz_dist(min_nnz, max_nnz);
    std::uniform_int_distribution<int> col_dist(0, cols - 1);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    std::vector<int> row_ptrs(rows + 1, 0);
    std::vector<int> col_indices;
    std::vector<float> values;
    for (int i = 0; i < rows; i++) {
        int row_nnz = nnz_dist(rng);
        row_ptrs[i + 1] = row_ptrs[i] + row_nnz;
        for (int j = 0; j < row_nnz; j++) {
            col_indices.push_back(col_dist(rng));
            values.push_back(val_dist(rng));
        }
    }

    const int nnz = row_ptrs[rows];
    CSRMatrix* csr = csr_create(rows, cols, nnz);
    if (!csr)
        return nullptr;
    for (int i = 0; i <= rows; i++)
        csr->row_ptrs[i] = row_ptrs[i];
    for (int i = 0; i < nnz; i++) {
        csr->col_indices[i] = col_indices[i];
        csr->values[i] = values[i];
    }
    return csr;
}

void run_kernel(const char* case_name, const CSRMatrix* csr, SpMVConfig::KernelType kernel,
                const char* kernel_name, const CudaBuffer<float>& d_x, CudaBuffer<float>& d_y,
                int repeats) {
    SpMVConfig config(kernel, DEFAULT_BLOCK_SIZE);

    // Warmup.
    for (int i = 0; i < 3; i++) {
        spmv_csr(csr, d_x.get(), d_y.get(), &config, csr->num_cols);
    }

    std::vector<float> timings;
    timings.reserve(repeats);
    float gflops = 0.0f, bandwidth = 0.0f;
    for (int i = 0; i < repeats; i++) {
        SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, csr->num_cols);
        if (spmv_result_error(result) != SpMVError::SUCCESS) {
            std::printf("%-14s %-12s FAILED: %s\n", case_name, kernel_name,
                        spmv_error_string(spmv_result_error(result)));
            return;
        }
        timings.push_back(result.elapsed_ms);
        gflops = result.gflops;
        bandwidth = result.bandwidth_gb_s;
    }

    std::sort(timings.begin(), timings.end());
    const float median_ms = timings[timings.size() / 2];
    std::printf("%-14s %-12s %10.4f ms %10.1f GFLOP/s %10.1f GB/s\n", case_name, kernel_name,
                median_ms, gflops, bandwidth);
}

void bench_case(const char* name, CSRMatrix* csr, int repeats) {
    if (!csr) {
        std::fprintf(stderr, "matrix setup failed for %s\n", name);
        return;
    }
    if (csr_to_gpu(csr) != 0) {
        std::fprintf(stderr, "csr_to_gpu failed for %s\n", name);
        csr_destroy(csr);
        return;
    }

    std::vector<float> x(static_cast<size_t>(csr->num_cols), 1.0f);
    CudaBuffer<float> d_x(x.size());
    CudaBuffer<float> d_y(static_cast<size_t>(csr->num_rows));
    d_x.copyFromHost(x.data(), x.size());

    SpMVConfig auto_config = spmv_auto_config(csr);
    const char* auto_name = auto_config.kernel_type == SpMVConfig::SCALAR_CSR   ? "scalar"
                            : auto_config.kernel_type == SpMVConfig::VECTOR_CSR ? "vector"
                                                                                : "merge-path";
    std::printf("%-14s auto-config selected: %s (block %d)\n", name, auto_name,
                auto_config.block_size);

    run_kernel(name, csr, SpMVConfig::SCALAR_CSR, "scalar", d_x, d_y, repeats);
    run_kernel(name, csr, SpMVConfig::VECTOR_CSR, "vector", d_x, d_y, repeats);
    run_kernel(name, csr, SpMVConfig::MERGE_PATH, "merge-path", d_x, d_y, repeats);

    csr_destroy(csr);
}

}  // namespace

int main(int argc, char** argv) {
    const int repeats = (argc > 2) ? std::atoi(argv[2]) : 50;

    std::printf("%-14s %-12s %13s %15s %13s\n", "case", "kernel", "median", "throughput",
                "bandwidth");

    if (argc > 1) {
        CSRMatrix* csr = csr_create(0, 0, 0);
        const int status = csr_read_matrix_market(csr, argv[1]);
        if (status != 0) {
            std::fprintf(stderr, "failed to load %s: %s\n", argv[1],
                         spmv_error_string(static_cast<SpMVError>(status)));
            csr_destroy(csr);
            return 1;
        }
        std::printf("loaded %s: %d x %d, nnz = %d\n", argv[1], csr->num_rows, csr->num_cols,
                    csr->nnz);
        bench_case("mtx", csr, repeats);
        return 0;
    }

    // Uniform short rows: scalar-friendly.
    bench_case("uniform-8", make_synthetic_csr(1 << 16, 1 << 14, 8, 8, 1), repeats);
    // Uniform medium rows: vector-friendly.
    bench_case("uniform-32", make_synthetic_csr(1 << 15, 1 << 14, 32, 32, 2), repeats);
    // Heavy skew: a few very long rows among short ones, merge-path territory.
    bench_case("skewed", make_synthetic_csr(1 << 15, 1 << 15, 1, 1 << 12, 3), repeats);

    return 0;
}
