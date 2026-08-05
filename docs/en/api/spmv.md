# SpMV Computation

Core SpMV computation APIs including kernel types, configuration, and execution.

## Kernel Types

```cpp
enum KernelType {
    SCALAR_CSR,  // 1 thread/row - for very sparse matrices
    VECTOR_CSR,  // 1 warp/row - for uniform distribution
    MERGE_PATH,  // Load-balanced - for skewed matrices
    ELL_KERNEL   // ELL format specific
};
```

## Configuration

### SpMVConfig

```cpp
struct SpMVConfig {
    KernelType kernel_type;
    int block_size;     // CUDA block size (default: 256)
    bool enable_timing; // default: true
};
```

With `enable_timing = true` (the default), `spmv_csr`/`spmv_ell` block until
`d_y` is complete and populate the timing fields of `SpMVResult`. Set it to
`false` to only enqueue the kernel on the stream — no CUDA events, no
synchronization — so calls can be pipelined; synchronize the stream yourself
before reading `d_y`. In async mode the timing fields stay zero and only
synchronous launch failures are reported.

### SpMVThresholds

Customizable thresholds for automatic kernel selection:

```cpp
struct SpMVThresholds {
    float avg_nnz_threshold;  // Default: 4.0
    float skewness_threshold; // Default: 10.0
};
```

### SpMVResult

```cpp
struct SpMVResult {
    float* y;              // Output vector (device pointer)
    float elapsed_ms;      // Execution time (ms)
    float gflops;          // Performance (GFLOPS)
    float bandwidth_gb_s;  // Memory bandwidth (GB/s), derived from elapsed_ms
    int error_code;        // 0 = success, negative = error
};

// Typed accessor for error_code
SpMVError spmv_result_error(const SpMVResult& result);
```

## Core Functions

### Automatic Configuration

```cpp
// Auto-select a CSR kernel configuration from matrix statistics
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// Configuration for ELL matrices (single kernel)
SpMVConfig spmv_auto_config_ell(const ELLMatrix* A);
```

### CSR SpMV

```cpp
// GPU SpMV with CSR format
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 for auto-detect
                    cudaStream_t stream = nullptr);
```

### ELL SpMV

```cpp
// GPU SpMV with ELL format
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);
```

### CPU Reference

```cpp
// CPU reference implementations (for validation).
// Return 0 on success, negative error code on invalid input.
int spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
int spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);
```

### Threshold Management

```cpp
SpMVThresholds spmv_get_thresholds();
void spmv_set_thresholds(const SpMVThresholds& thresholds);
```

## Error Handling

```cpp
enum class SpMVError {
    SUCCESS = 0,             // Success
    INVALID_DIMENSION = -1,  // Dimension mismatch
    CUDA_MALLOC = -2,        // GPU memory allocation failed
    CUDA_MEMCPY = -3,        // Memory copy failed
    KERNEL_LAUNCH = -4,      // Kernel launch failed
    INVALID_FORMAT = -5,     // Invalid format
    FILE_IO = -6,            // File IO error
    OUT_OF_MEMORY = -7,      // Out of memory
    INVALID_ARGUMENT = -8    // Invalid argument
};

const char* spmv_error_string(SpMVError err);
```

## Complete Example

```cpp
#include <spmv/spmv.h>

int main() {
    // 1. Create CSR matrix
    CSRMatrix* csr = csr_create(1000, 1000, 10000);
    // ... fill data ...
    csr_to_gpu(csr);

    // 2. Prepare vectors
    CudaBuffer<float> d_x(1000), d_y(1000);

    // 3. Auto-configure and execute
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config);

    // 4. Check result
    if (spmv_result_error(result) != SpMVError::SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_error_string(spmv_result_error(result)));
        return 1;
    }

    printf("Time: %.3f ms\n", result.elapsed_ms);
    printf("Bandwidth: %.1f GB/s\n", result.bandwidth_gb_s);

    csr_destroy(csr);
    return 0;
}
```

## Headers

```cpp
#include <spmv/csr_matrix.h>   // CSR matrix
#include <spmv/cuda_buffer.h>  // RAII memory management
#include <spmv/ell_matrix.h>   // ELL matrix
#include <spmv/market_io.h>    // Matrix Market file reader
#include <spmv/spmv.h>         // Main interface + SpMV computation
```
