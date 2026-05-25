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
    int block_size;    // CUDA block size (default: 256)
    bool use_texture;  // Whether to use texture cache
};
```

### SpMVThresholds

Customizable thresholds for automatic kernel selection:

```cpp
struct SpMVThresholds {
    float avg_nnz_threshold;     // Default: 4.0
    float skewness_threshold;    // Default: 10.0
    int texture_cols_threshold;  // Default: 10000
};
```

### SpMVExecutionContext

Optional context for texture cache reuse across iterations:

```cpp
class SpMVExecutionContext {
   public:
    SpMVExecutionContext();
    ~SpMVExecutionContext();

    void reset();              // Reset and release texture object
    bool is_texture_bound();   // Query whether texture is currently bound

    // Copy disabled, move allowed
    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext& operator=(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext(SpMVExecutionContext&&) noexcept;
    SpMVExecutionContext& operator=(SpMVExecutionContext&&) noexcept;
};
```

### SpMVResult

```cpp
struct SpMVResult {
    float* y;              // Output vector (device pointer)
    float elapsed_ms;      // Execution time (ms)
    float gflops;          // Performance (GFLOPS)
    float bandwidth_gb_s;  // Memory bandwidth (GB/s)
    int error_code;        // 0 = success, negative = error
};
```

## Core Functions

### Automatic Configuration

```cpp
// Auto-select optimal kernel configuration
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

### CSR SpMV

```cpp
// GPU SpMV with CSR format
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 for auto-detect
                    SpMVExecutionContext* context = nullptr);
```

### ELL SpMV

```cpp
// GPU SpMV with ELL format
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);
```

### CPU Reference

```cpp
// CPU reference implementations (for validation)
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);
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
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config);

    // 4. Check result
    if (result.error_code != 0) {
        fprintf(stderr, "Error: %d\n", result.error_code);
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
#include <spmv/spmv.h>         // Main interface + SpMV computation
```
