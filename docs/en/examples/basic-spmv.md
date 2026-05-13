# Basic SpMV Example

Complete example demonstrating GPU SpMV computation.

## Full Example

```cpp
#include <spmv/spmv.h>
#include <cstdio>
#include <random>

int main() {
    // ===== 1. Matrix Setup =====
    const int N = 10000;      // Matrix dimension
    const int NNZ = 100000;   // Non-zeros

    // Create CSR matrix
    CSRMatrix* csr = csr_create(N, N, NNZ);

    // Fill with random sparse data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> col_dist(0, N - 1);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);

    // Build dense representation first (for simplicity)
    std::vector<float> dense(N * N, 0.0f);
    for (int i = 0; i < NNZ; i++) {
        int row = i * N / NNZ;
        int col = col_dist(rng);
        dense[row * N + col] = val_dist(rng);
    }

    // Convert to CSR
    csr_from_dense(csr, dense.data(), N, N);

    // ===== 2. GPU Transfer =====
    csr_to_gpu(csr);

    // ===== 3. Vector Setup =====
    // Input vector x = [1, 1, 1, ...]
    std::vector<float> h_x(N, 1.0f);
    CudaBuffer<float> d_x(N), d_y(N);

    cudaMemcpy(d_x.data(), h_x.data(), N * sizeof(float),
               cudaMemcpyHostToDevice);

    // ===== 4. Auto Configuration =====
    SpMVConfig config = spmv_auto_config(csr);

    // Print selected kernel
    const char* kernel_names[] = {
        "Scalar CSR", "Vector CSR", "Merge Path", "ELL Kernel"
    };
    printf("Selected kernel: %s\n", kernel_names[config.kernel_type]);

    // ===== 5. Execute SpMV =====
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config);

    // ===== 6. Check Results =====
    if (result.error_code != 0) {
        fprintf(stderr, "Error: %d\n", result.error_code);
        csr_destroy(csr);
        return 1;
    }

    printf("Time: %.3f ms\n", result.elapsed_ms);
    printf("Bandwidth: %.1f GB/s\n", result.bandwidth_gb_s);
    printf("GFLOPS: %.1f\n", result.gflops);

    // ===== 7. Verify (optional) =====
    std::vector<float> h_y_gpu(N), h_y_cpu(N);
    cudaMemcpy(h_y_gpu.data(), d_y.data(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // CPU reference
    spmv_cpu_csr(csr, h_x.data(), h_y_cpu.data());

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = std::abs(h_y_gpu[i] - h_y_cpu[i]);
        max_error = std::max(max_error, error);
    }
    printf("Max error: %.6e\n", max_error);

    // ===== 8. Cleanup =====
    csr_destroy(csr);
    return 0;
}
```

## Compilation

```bash
nvcc -o spmv_example spmv_example.cpp \
    -I./include \
    -L./build-release -lgpu_spmv \
    -lcudart
```

## Expected Output

```
Selected kernel: Vector CSR
Time: 2.345 ms
Bandwidth: 68.5 GB/s
GFLOPS: 85.6
Max error: 1.234567e-05
```

## Key Points

1. **Use `spmv_auto_config()`** for automatic kernel selection
2. **Use `CudaBuffer`** for RAII memory management
3. **Verify with CPU reference** for correctness testing
4. **Check `error_code`** for error handling
