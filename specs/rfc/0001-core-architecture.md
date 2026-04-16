# RFC 0001: Core Architecture - GPU SpMV (Sparse Matrix-Vector Multiplication)

> **Version**: v1.0.0
> **Status**: IMPLEMENTED
> **Last Updated**: 2025-04-16
> **Authors**: GPU SpMV Team

---

## Abstract

This document describes the CUDA-based Sparse Matrix-Vector Multiplication (SpMV) implementation. The system adopts a layered architecture comprising a storage layer (CSR/ELL formats), a compute layer (multiple optimized kernels), and an application layer (PageRank algorithm).

### Design Goals

| Goal | Description | Validation Method |
|------|-------------|-------------------|
| **Correctness** | SpMV results match the CPU reference implementation | Property-based testing |
| **Performance** | Maximize GPU bandwidth utilization | Benchmarking |
| **Scalability** | Support for large-scale sparse matrices | Large matrix testing |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ PageRank        │    │ Benchmark Suite                  │ │
│  │ Algorithm       │    │ - Statistical Analysis           │ │
│  │ - Iterative     │    │ - GPU vs CPU Comparison          │ │
│  │ - Convergence   │    │ - JSON Export                    │ │
│  └────────┬────────┘    └──────────────┬──────────────────┘ │
└───────────┼────────────────────────────┼────────────────────┘
            │                            │
┌───────────┼────────────────────────────┼────────────────────┐
│           ▼         Compute Layer      ▼                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Kernel Selector                          │   │
│  │  spmv_auto_config() - Matrix Statistics → Kernel     │   │
│  └─────┬─────────────┬──────────────┬──────────────┬────┘   │
│        │             │              │              │        │
│   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   │
│   │ Scalar  │   │ Vector  │   │  Merge  │   │   ELL   │   │
│   │  CSR    │   │  CSR    │   │  Path   │   │ Kernel  │   │
│   │         │   │         │   │         │   │         │   │
│   │ 1 thread│   │ 1 warp  │   │ Perfect │   │ Column- │   │
│   │ /row    │   │ /row    │   │ balance │   │ major   │   │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘   │
└────────┼─────────────┼─────────────┼─────────────┼─────────┘
         │             │             │             │
┌────────▼─────────────▼─────────────▼─────────────▼─────────┐
│                      Storage Layer                          │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   CSR Format     │◄───────►│   ELL Format     │         │
│  │ - values[]       │  Conv.  │ - values[][]     │         │
│  │ - col_indices[]  │         │ - col_indices[][]│         │
│  │ - row_ptrs[]     │         │ (column-major)   │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
└───────────┼────────────────────────────┼───────────────────┘
            │                            │
┌───────────▼────────────────────────────▼───────────────────┐
│                   Memory Management                         │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │ CudaBuffer<T>   │    │ Texture Cache                   ││
│  │ - RAII Pattern  │    │ - Input Vector Caching          ││
│  │ - Auto Cleanup  │    │ - SpMVExecutionContext          ││
│  └─────────────────┘    └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Components and Interfaces

### 1. Sparse Matrix Storage Formats

#### CSR (Compressed Sparse Row) Format

```cpp
struct CSRMatrix {
    int num_rows;           // Number of rows in the matrix
    int num_cols;           // Number of columns in the matrix
    int nnz;                // Total number of non-zero elements

    float* values;          // Non-zero element values array [nnz]
    int* col_indices;       // Column indices array [nnz]
    int* row_ptrs;          // Row pointers array [num_rows + 1]

    // Device-side (GPU) pointers
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

**Memory Layout Example**:
```
Sparse Matrix:              CSR Storage:
| 1 0 2 0 |                 values:      [1, 2, 3, 4, 5]
| 0 3 4 0 |      =>         col_indices: [0, 2, 1, 2, 3]
| 0 0 0 5 |                 row_ptrs:    [0, 2, 4, 5]
                            (Row 0: indices 0-1, 2 elements)
                            (Row 1: indices 2-3, 2 elements)
                            (Row 2: index 4,    1 element)
```

#### ELL (ELLPACK) Format

```cpp
struct ELLMatrix {
    int num_rows;           // Number of rows in the matrix
    int num_cols;           // Number of columns in the matrix
    int max_nnz_per_row;    // Maximum non-zero elements per row
    int nnz;                // Actual total number of non-zero elements

    // Column-major storage for coalesced access
    float* values;          // Values array [num_rows * max_nnz_per_row]
    int* col_indices;       // Column indices [-1 indicates padding]

    float* d_values;        // Device-side (GPU) pointers
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

**Column-Major Storage Explanation**:
```
Sparse Matrix (max_nnz_per_row = 2):
| 1 0 2 |     Row 0: [1, 2] columns [0, 2]
| 3 4 0 | =>  Row 1: [3, 4] columns [0, 1]
| 5 0 0 |     Row 2: [5, -] columns [0, -]

Column-major storage:
values:     [1, 3, 5, 2, 4, 0]     // Stored by column
col_indices: [0, 0, 0, 2, 1, -1]   // -1 indicates padding

GPU access: Thread i accesses values[k*num_rows + i], contiguous addresses!
```

---

### 2. SpMV Kernel Interfaces

```cpp
// Unified SpMV configuration
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,     // One thread processes one row
        VECTOR_CSR,     // One warp processes one row
        MERGE_PATH,     // Evenly distributed workload
        ELL_KERNEL      // ELL format dedicated kernel
    };

    KernelType kernel_type;
    int block_size;         // CUDA block size
    bool use_texture;       // Whether to use texture cache
};

// Computation result
struct SpMVResult {
    float* y;               // Output vector (GPU)
    float elapsed_ms;       // Execution time
    float gflops;           // Computation throughput
    float bandwidth_gb_s;   // Bandwidth utilization
    int error_code;         // 0 = success
};

// Core SpMV function
SpMVResult spmv_csr(
    const CSRMatrix* A,
    const float* d_x,
    float* d_y,
    const SpMVConfig* config = nullptr,
    int vec_size = -1,
    SpMVExecutionContext* context = nullptr
);

SpMVResult spmv_ell(
    const ELLMatrix* A,
    const float* d_x,
    float* d_y,
    const SpMVConfig* config = nullptr,
    int vec_size = -1,
    SpMVExecutionContext* context = nullptr
);

// Automatically select the optimal kernel
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

---

### 3. CUDA Kernel Design

#### 3.1 Scalar CSR Kernel

**Strategy**: One thread processes one row

**Applicable Scenario**: `avg_nnz_per_row < 4` (extremely sparse matrices)

**Advantages**: Simple implementation, no synchronization overhead

**Disadvantages**: Long rows cause warp idle time

```cpp
__global__ void spmv_csr_scalar(
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
        for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}
```

#### 3.2 Vector CSR Kernel

**Strategy**: One warp (32 threads) collaboratively processes one row

**Applicable Scenario**: `skewness < 10` (uniform row length distribution)

**Advantages**: Efficient warp-level reduction, good coalesced access

**Disadvantages**: Long rows still create bottlenecks

```cpp
__global__ void spmv_csr_vector(
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

        // Threads within a warp collaborate to process one row
        for (int j = row_ptrs[warp_id] + lane_id;
             j < row_ptrs[warp_id + 1]; j += 32) {
            sum += values[j] * x[col_indices[j]];
        }

        // Warp-level reduction (using shuffle instructions)
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) y[warp_id] = sum;
    }
}
```

#### 3.3 Merge Path Kernel

**Strategy**: Treats the row pointer and non-zero element sequences as two ordered paths, finding uniform split points via binary search

**Applicable Scenario**: `skewness >= 10` (highly non-uniform distribution)

**Advantages**: Perfect load balancing

**Disadvantages**: Complex implementation

```cpp
struct MergeCoordinate {
    int row;    // Current row
    int nz;     // Current non-zero element position
};

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

    return {x_min, diagonal - x_min};
}
```

**Merge Path Visualization**:
```
Row pointer sequence:   [0, 2, 5, 7, 10]    (4 rows, 10 non-zero elements total)
Non-zero element seq:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Merge Path perspective:
         row:  0   1   2   3   4
              [0]─[2]─[5]─[7]─[10]
               │╲  │╲  │╲  │╲
              [0][1][2][3][4][5][6][7][8][9]  (nz)
               0   1   2   3   4   5   6   7   8   9

Uniform partitioning: Diagonal cuts, each thread handles an equal amount of work
```

#### 3.4 ELL Kernel

**Strategy**: Column-major storage, naturally coalesced access

**Applicable Scenario**: Row lengths close to `max_nnz_per_row`

**Advantages**: Fully coalesced memory access, no conditional branching

**Disadvantages**: Memory waste due to padding

```cpp
__global__ void spmv_ell(
    int num_rows,
    int max_nnz_per_row,
    const int* col_indices,    // Column-major
    const float* values,
    const float* x,
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int k = 0; k < max_nnz_per_row; k++) {
            int idx = k * num_rows + row;  // Column-major index
            int col = col_indices[idx];
            if (col >= 0) {  // -1 indicates padding
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
```

---

### 4. Kernel Selection Strategy

```
                    Matrix Feature Analysis
                         │
                         ▼
         ┌───────────────────────────────┐
         │   avg_nnz_per_row < 4 ?       │
         └───────────────────────────────┘
                 │           │
                Yes          No
                 │           │
                 ▼           ▼
         ┌───────────┐  ┌───────────────────┐
         │   Scalar  │  │  skewness < 10 ?  │
         │    CSR    │  └───────────────────┘
         └───────────┘          │           │
                               Yes          No
                                │           │
                                ▼           ▼
                        ┌───────────┐ ┌───────────┐
                        │  Vector   │ │   Merge   │
                        │    CSR    │ │   Path    │
                        └───────────┘ └───────────┘
```

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* A) {
    SpMVConfig config;
    config.block_size = DEFAULT_BLOCK_SIZE;  // 256
    config.use_texture = (A->num_cols > TEXTURE_CACHE_THRESHOLD_COLS);

    // Compute row length statistics
    CSRStats stats = csr_compute_stats(A);

    // Selection strategy
    if (stats.avg_nnz_per_row < 4) {
        config.kernel_type = SpMVConfig::SCALAR_CSR;
    } else if (stats.skewness < 10) {
        config.kernel_type = SpMVConfig::VECTOR_CSR;
    } else {
        config.kernel_type = SpMVConfig::MERGE_PATH;
    }

    return config;
}
```

---

### 5. Bandwidth Optimization

#### 5.1 Column-Major Storage (ELL Format)

```
Row-major access pattern (poor):
Thread:   T0      T1      T2
          ↓       ↓       ↓
Address: [row0,k0][row1,k0][row2,k0]  ← Discontiguous!
        [base+0] [base+max_nnz] [base+2*max_nnz]

Column-major access pattern (good):
Thread:   T0      T1      T2
          ↓       ↓       ↓
Address: [row0,k0][row1,k0][row2,k0]  ← Contiguous!
        [base+0] [base+1]   [base+2]
```

#### 5.2 Texture Cache

```cpp
// Use SpMVExecutionContext to reuse texture objects
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < iterations; i++) {
    // Texture object is only created on the first call
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
// Texture object is automatically destroyed when context is destructed
```

#### 5.3 Warp-Level Reduction

```cpp
// Traditional Shared Memory reduction (risk of bank conflicts)
__shared__ float sdata[32];
sdata[lane_id] = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    sdata[lane_id] += sdata[lane_id + offset];  // Possible conflicts
}

// Shuffle reduction (no bank conflicts)
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);  // Fully parallel
}
```

---

### 6. PageRank Algorithm

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // Damping factor
    float tolerance = 1e-6f;        // Convergence threshold
    int max_iterations = 100;       // Maximum number of iterations
};

struct PageRankResult {
    float* ranks;           // Ranking scores array [num_nodes]
    int iterations;         // Actual number of iterations
    float final_residual;   // Final residual
    bool converged;         // Whether convergence was achieved
    int error_code;         // Error code
};

// PageRank iteration: r_{k+1} = d * A * r_k + (1-d) / n
// where A is the column-normalized adjacency matrix
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,
    const PageRankConfig* config
);
```

---

### 7. Error Handling

#### Error Code Definitions

```cpp
enum class SpMVError {
    SUCCESS = 0,
    INVALID_DIMENSION = -1,   // Matrix or vector dimension mismatch
    CUDA_MALLOC = -2,         // GPU memory allocation failure
    CUDA_MEMCPY = -3,         // GPU memory copy failure
    KERNEL_LAUNCH = -4,       // CUDA kernel launch/execution failure
    INVALID_FORMAT = -5,      // Invalid sparse matrix format
    FILE_IO = -6,             // File read/write failure
    OUT_OF_MEMORY = -7,       // Host/device memory exhaustion
    INVALID_ARGUMENT = -8     // Invalid argument
};

const char* spmv_error_string(SpMVError err);
```

#### CUDA Check Macros

```cpp
#define CUDA_CHECK_MALLOC(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return static_cast<int>(SpMVError::CUDA_MALLOC); \
    } \
} while(0)

#define CUDA_CHECK_MEMCPY(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return static_cast<int>(SpMVError::CUDA_MEMCPY); \
    } \
} while(0)
```

#### RAII Resource Management

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();

    // Copy disabled
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Move allowed
    CudaBuffer(CudaBuffer&& other) noexcept;

    T* get();
    const T* get() const;
    size_t size() const;

    void copyFromHost(const T* host_ptr, size_t count);
    void copyToHost(T* host_ptr, size_t count);
    void memset(int value);
    void fill(const T& value);

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};
```

---

## Correctness Properties

### Property Test List

| ID | Property Name | Validation Requirements |
|----|---------------|-------------------------|
| P1 | CSR Dense-to-Sparse Round Trip | 1.2 |
| P2 | CSR Element Lookup Correctness | 1.3 |
| P3 | CSR Serialization Round Trip | 1.5 |
| P4 | ELL Dense-to-Sparse Round Trip | 2.2 |
| P5 | ELL Padding Correctness | 2.3 |
| P6 | ELL Column-Major Layout | 2.4 |
| P7 | ELL Serialization Round Trip | 2.5 |
| P8 | SpMV CSR Correctness | 3.1, 3.3 |
| P9 | SpMV ELL Correctness | 3.2, 3.3 |
| P10 | SpMV Dimension Validation | 3.5, 8.5 |
| P11 | Kernel Selector Validity | 4.5 |
| P12 | Bandwidth Metrics Validity | 5.5 |
| P13 | Benchmark Metrics Completeness | 6.1, 6.3 |
| P14 | Benchmark JSON Round Trip | 6.5 |
| P15 | PageRank Score Invariants | 7.1, 7.2 |
| P16 | PageRank Top-K Ordering | 7.5 |

### Property Test Template

```cpp
// **Property 8: SpMV CSR Correctness**
// **Validates: Requirements 3.1, 3.3**
TEST(SpMVPropertyTest, CSRCorrectnessProperty) {
    for (int iter = 0; iter < 100; iter++) {
        auto matrix = generator.generate();
        auto x = generate_random_vector(matrix->num_cols);

        // GPU computation
        SpMVResult gpu_result = spmv_csr(matrix, d_x, d_y, &config);

        // CPU reference
        spmv_cpu_csr(matrix, x.data(), y_cpu.data());

        // Verify relative error
        for (int i = 0; i < matrix->num_rows; i++) {
            if (y_cpu[i] != 0) {
                EXPECT_LT(abs(y_gpu[i] - y_cpu[i]) / abs(y_cpu[i]), 1e-6);
            }
        }
    }
}
```

---

## Testing Strategy

### Test Types

| Type | Framework | Purpose |
|------|-----------|---------|
| Unit Testing | Google Test | Test specific examples and edge cases |
| Property-based Testing | Google Test + Random Generation | Validate general properties |
| Performance Testing | CUDA Events | Measure execution time and bandwidth |

### Test Matrix Generator

```cpp
struct SparseMatrixGenerator {
    int min_rows = 1, max_rows = 1000;
    int min_cols = 1, max_cols = 1000;
    float min_density = 0.001, max_density = 0.3;

    enum RowDistribution {
        UNIFORM,        // Similar number of non-zero elements per row
        POWER_LAW,      // Power law distribution (simulates real graphs)
        EXTREME_SKEW    // Extremely non-uniform
    };

    CSRMatrix* generate(RowDistribution dist = UNIFORM);
};
```

### Edge Case Handling

| Case | Handling |
|------|----------|
| Empty matrix (0 rows or 0 columns) | Return empty result vector |
| All-zero row | SpMV processes normally, result is 0 |
| Single element matrix | Normal processing |
| Extremely large matrix (exceeds GPU memory) | Return OUT_OF_MEMORY error |
| NaN/Inf input values | Propagated to output (IEEE 754 semantics) |
