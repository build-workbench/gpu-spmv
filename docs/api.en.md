-- -layout : default title : API Reference nav_order : 3 permalink : / api.en lang : en-- -

    <p align = "right"><a href = "api">🇨🇳 简体中文</ a></ p>
#API Reference
    { :.no_toc}

    Complete GPU SpMV public API documentation.{: .fs-6 .fw-300
}

##Table of Contents {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

   ##Headers

```cpp
#include <spmv/benchmark.h>    // Benchmarking
#include <spmv/csr_matrix.h>   // CSR matrix
#include <spmv/cuda_buffer.h>  // RAII memory management
#include <spmv/ell_matrix.h>   // ELL matrix
#include <spmv/pagerank.h>     // PageRank
#include <spmv/spmv.h>         // Main interface + SpMV computation
```

   -- -

   ##Error Handling

```cpp enum class SpMVError {
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

const char* spmv_error_string(SpMVError err);  // Get error description string
```

    -- -

    ##CSR Matrix

    ## #Data Structure

```cpp struct CSRMatrix {
    int num_rows;  // Number of rows
    int num_cols;  // Number of columns
    int nnz;       // Total non-zero elements

    float* values;     // Non-zero values [nnz]
    int* col_indices;  // Column indices [nnz]
    int* row_ptrs;     // Row pointers [num_rows + 1]

    // GPU device pointers
    float* d_values;     // Device values
    int* d_col_indices;  // Device column indices
    int* d_row_ptrs;     // Device row pointers

    // Memory ownership flags
    bool owns_host_memory;    // Owns host memory
    bool owns_device_memory;  // Owns device memory
};
```

    ## #Core Functions

```cpp
    // Create and destroy
    CSRMatrix*
    csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* mat);

// Data conversion
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
int csr_to_dense(const CSRMatrix* csr, float* dense);

// GPU data transfer
int csr_to_gpu(CSRMatrix* mat);
int csr_from_gpu(CSRMatrix* mat);
void csr_free_gpu(CSRMatrix* mat);

// Element access
float csr_get_element(const CSRMatrix* mat, int row, int col);

// Serialization
int csr_serialize(const CSRMatrix* mat, const char* filename);
int csr_deserialize(CSRMatrix* mat, const char* filename);  // Note: in-place style

// Statistics and validation
CSRStats csr_compute_stats(const CSRMatrix* mat);
bool csr_validate(const CSRMatrix* mat);
```

    ## #CSRStats Structure

```cpp struct CSRStats {
    float avg_nnz_per_row;  // Average non-zeros per row
    int max_nnz_per_row;    // Maximum non-zeros in any row
    int min_nnz_per_row;    // Minimum non-zeros in any row
    float skewness;         // Skewness: max / (min + 1)
};
```

    -- -

    ##ELL Matrix

    ## #Data Structure

```cpp struct ELLMatrix {
    int num_rows;         // Number of rows
    int num_cols;         // Number of columns
    int max_nnz_per_row;  // Max non-zeros per row (determines padding)
    int nnz;              // Actual total non-zero count

    // Column-major storage: values[k * num_rows + row]
    float* values;     // Values [num_rows * max_nnz_per_row]
    int* col_indices;  // Column indices, -1 indicates padding

    // GPU device pointers
    float* d_values;
    int* d_col_indices;

    // Memory ownership flags
    bool owns_host_memory;
    bool owns_device_memory;
};
```

    ## #Core Functions

```cpp
    // Create and destroy
    ELLMatrix*
    ell_create(int rows, int cols, int max_nnz_per_row);
void ell_destroy(ELLMatrix* mat);

// Data conversion
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
int ell_to_dense(const ELLMatrix* ell, float* dense);

// GPU data transfer
int ell_to_gpu(ELLMatrix* mat);
int ell_from_gpu(ELLMatrix* mat);
void ell_free_gpu(ELLMatrix* mat);

// Element access
float ell_get_element(const ELLMatrix* mat, int row, int col);

// Serialization
int ell_serialize(const ELLMatrix* mat, const char* filename);
int ell_deserialize(ELLMatrix* mat, const char* filename);

// Validation
bool ell_validate(const ELLMatrix* mat);
```

    -- -

    ##SpMV Computation

    ## #Kernel Types

```cpp enum KernelType {
        SCALAR_CSR,  // 1 thread/row - best for very sparse matrices
        VECTOR_CSR,  // 1 warp/row - best for uniform distribution
        MERGE_PATH,  // Load balanced - best for skewed matrices
        ELL_KERNEL   // ELL format specific
    };
```

    ## #Configuration Structures

```cpp struct SpMVConfig {
    KernelType kernel_type;
    int block_size;    // CUDA block size (default: 256)
    bool use_texture;  // Use texture cache
};

// Threshold configuration (for auto-selection)
struct SpMVThresholds {
    float avg_nnz_threshold;     // Default: 4.0
    float skewness_threshold;    // Default: 10.0
    int texture_cols_threshold;  // Default: 10000
};
```

### Execution Context (Optional, for texture cache reuse)

```cpp
struct SpMVExecutionContext {
    cudaTextureObject_t tex_x;  // Texture object
    const float* cached_x;      // Cached x pointer
    size_t cached_x_length;     // Cached x length
    bool texture_enabled;       // Whether texture is enabled

    SpMVExecutionContext();
    ~SpMVExecutionContext();

    void reset();  // Reset and free texture object

    // Copy disabled, move enabled
    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext(SpMVExecutionContext&&) noexcept;
};
```

    ## #Result Structure

```cpp struct SpMVResult {
    float* y;              // Output vector (device pointer)
    float elapsed_ms;      // Execution time (milliseconds)
    float gflops;          // Performance (GFLOPS)
    float bandwidth_gb_s;  // Memory bandwidth (GB/s)
    int error_code;        // 0 = success, negative = error
};
```

    ## #Core Functions

```cpp
        // Auto-select optimal configuration
        SpMVConfig
        spmv_auto_config(const CSRMatrix* A);

// CSR SpMV (GPU)
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 for auto-detect
                    SpMVExecutionContext* context = nullptr);

// ELL SpMV (GPU)
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);

// CPU reference implementations (for verification)
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

// Threshold configuration
SpMVThresholds spmv_get_thresholds();
void spmv_set_thresholds(const SpMVThresholds& thresholds);
```

    -- -

    ##RAII Memory Management

```cpp template <typename T>
    class CudaBuffer {
   public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();  // Automatically frees GPU memory

    T* data();
    const T* data() const;
    size_t size() const;

    // Copy disabled
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Move enabled
    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
};
```

    -- -

    ##PageRank

    ## #Configuration and Result

```cpp struct PageRankConfig {
    float damping_factor;  // Damping factor (default: 0.85)
    float tolerance;       // Convergence threshold (default: 1e-6)
    int max_iterations;    // Max iterations (default: 100)
};

struct PageRankResult {
    float* ranks;          // PageRank scores [num_nodes]
    int iterations;        // Actual iterations performed
    float final_residual;  // Final residual
    bool converged;        // Whether converged
    int error_code;        // 0 = success
};

struct TopKNode {
    int node_id;  // Node ID
    float rank;   // PageRank score
};
```

    ## #Core Functions

```cpp
        // Compute PageRank
        PageRankResult
        pagerank(const CSRMatrix* adj_matrix, const PageRankConfig* config = nullptr);

// Get Top-K nodes
void pagerank_top_k(const PageRankResult* result, int num_nodes, int k, TopKNode* top_k);

// Free result memory
void pagerank_free(PageRankResult* result);
```

    -- -

    ##Complete Example

```cpp
#include <spmv/spmv.h>

    int
    main() {
    // 1. Create CSR matrix
    CSRMatrix* csr = csr_create(1000, 1000, 10000);
    // ... fill data ...
    csr_to_gpu(csr);

    // 2. Prepare vectors
    CudaBuffer<float> d_x(1000), d_y(1000);

    // 3. Auto-config and execute
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

    -- -

    <div class = "text-center text-small" style = "margin-top: 3rem;">
    <p> More examples on<a href = "examples.en"> Examples</ a> page</ p></ div>
