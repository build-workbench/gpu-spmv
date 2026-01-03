# Design Document: GPU SpMV (稀疏矩阵向量乘法)

## Overview

本设计文档描述基于 CUDA 的稀疏矩阵向量乘法 (SpMV) 实现。系统采用分层架构，包含存储层（CSR/ELL 格式）、计算层（多种优化 Kernel）、和应用层（PageRank 算法）。

核心设计目标：
1. **正确性**：SpMV 结果与 CPU 参考实现一致
2. **性能**：最大化 GPU 带宽利用率，解决负载不均衡问题
3. **可扩展性**：支持大规模稀疏矩阵（千万级非零元素）

## Architecture

```mermaid
graph TB
    subgraph Application Layer
        PR[PageRank Algorithm]
        BM[Benchmark Suite]
    end
    
    subgraph Compute Layer
        KS[Kernel Selector]
        K1[Scalar CSR Kernel]
        K2[Vector CSR Kernel]
        K3[Merge Path Kernel]
        K4[ELL Kernel]
    end
    
    subgraph Storage Layer
        CSR[CSR Format]
        ELL[ELL Format]
        CONV[Format Converter]
    end
    
    subgraph Memory Management
        GPU[GPU Memory Manager]
        TEX[Texture Cache]
    end
    
    PR --> KS
    BM --> KS
    KS --> K1 & K2 & K3 & K4
    K1 & K2 & K3 --> CSR
    K4 --> ELL
    CSR & ELL --> GPU
    GPU --> TEX
end
```


## Components and Interfaces

### 1. 稀疏矩阵存储格式

#### CSR (Compressed Sparse Row) 格式

```cpp
struct CSRMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int nnz;                // 非零元素总数
    
    float* values;          // 非零元素值数组 [nnz]
    int* col_indices;       // 列索引数组 [nnz]
    int* row_ptrs;          // 行指针数组 [num_rows + 1]
    
    // GPU 端指针
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;
};

// 接口函数
CSRMatrix* csr_create(int rows, int cols, int nnz);
void csr_destroy(CSRMatrix* mat);
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
int csr_to_gpu(CSRMatrix* mat);
int csr_from_gpu(CSRMatrix* mat);
int csr_serialize(const CSRMatrix* mat, const char* filename);
int csr_deserialize(CSRMatrix* mat, const char* filename);
float csr_get_element(const CSRMatrix* mat, int row, int col);
```

#### ELL (ELLPACK) 格式

```cpp
struct ELLMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int max_nnz_per_row;    // 每行最大非零元素数
    
    // Column-major 存储以实现合并访问
    float* values;          // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;       // 列索引 [num_rows * max_nnz_per_row], -1 表示填充
    
    float* d_values;
    int* d_col_indices;
};

// 接口函数
ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row);
void ell_destroy(ELLMatrix* mat);
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
int ell_to_gpu(ELLMatrix* mat);
int ell_serialize(const ELLMatrix* mat, const char* filename);
int ell_deserialize(ELLMatrix* mat, const char* filename);
```


### 2. SpMV Kernel 接口

```cpp
// 统一的 SpMV 接口
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,     // 一个线程处理一行
        VECTOR_CSR,     // 一个 Warp 处理一行
        MERGE_PATH,     // 工作量均匀分配
        ELL_KERNEL      // ELL 格式专用
    };
    
    KernelType kernel_type;
    int block_size;         // CUDA block 大小
    bool use_texture;       // 是否使用纹理缓存
};

struct SpMVResult {
    float* y;               // 输出向量 (GPU)
    float elapsed_ms;       // 执行时间
    float gflops;           // 计算吞吐量
    float bandwidth_gb_s;   // 带宽利用率
    int error_code;         // 0 = 成功
};

// 核心 SpMV 函数
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y, 
                    const SpMVConfig* config);
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config);

// 自动选择最优 Kernel
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

### 3. CUDA Kernel 设计

#### Scalar CSR Kernel (基础版本)
```cpp
// 一个线程处理一行 - 简单但负载不均衡
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
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        y[row] = sum;
    }
}
```


#### Vector CSR Kernel (Warp 级并行)
```cpp
// 一个 Warp (32线程) 处理一行 - 更好的负载均衡
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
        int row_start = row_ptrs[warp_id];
        int row_end = row_ptrs[warp_id + 1];
        
        // Warp 内线程协作处理一行
        for (int j = row_start + lane_id; j < row_end; j += 32) {
            sum += values[j] * x[col_indices[j]];
        }
        
        // Warp 级归约
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            y[warp_id] = sum;
        }
    }
}
```

#### Merge Path Kernel (高级负载均衡)
```cpp
// Merge Path 算法 - 工作量完全均匀分配
// 将 (row_ptrs, 0..nnz) 视为两个有序序列，找到均匀分割点

struct MergeCoordinate {
    int row;    // 当前行
    int nz;     // 当前非零元素位置
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


#### ELL Kernel
```cpp
// ELL 格式 - 天然适合 GPU 合并访问
__global__ void spmv_ell(
    int num_rows,
    int max_nnz_per_row,
    const int* col_indices,    // Column-major [num_rows * max_nnz_per_row]
    const float* values,
    const float* x,
    float* y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        for (int j = 0; j < max_nnz_per_row; j++) {
            int idx = j * num_rows + row;  // Column-major 索引
            int col = col_indices[idx];
            if (col >= 0) {  // -1 表示填充
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
```

### 4. 带宽优化策略

```cpp
// 纹理内存绑定 (用于输入向量 x)
texture<float, 1, cudaReadModeElementType> tex_x;

__device__ float fetch_x(int idx) {
    return tex1Dfetch(tex_x, idx);
}

// 性能度量
struct BandwidthMetrics {
    float theoretical_bandwidth_gb_s;  // GPU 理论峰值带宽
    float achieved_bandwidth_gb_s;     // 实际达到的带宽
    float efficiency;                  // 带宽利用率 = achieved / theoretical
    
    // 计算方法: 
    // bytes_transferred = nnz * (sizeof(float) + sizeof(int))  // values + col_indices
    //                   + (num_rows + 1) * sizeof(int)         // row_ptrs
    //                   + num_cols * sizeof(float)             // x vector
    //                   + num_rows * sizeof(float)             // y vector
    // bandwidth = bytes_transferred / elapsed_time
};

BandwidthMetrics compute_bandwidth(const CSRMatrix* A, float elapsed_ms);
```


### 5. Kernel 选择器

```cpp
// 基于矩阵特征自动选择最优 Kernel
SpMVConfig spmv_auto_config(const CSRMatrix* A) {
    SpMVConfig config;
    config.block_size = 256;
    config.use_texture = (A->num_cols > 10000);  // 大向量使用纹理缓存
    
    // 计算行长度统计
    float avg_nnz_per_row = (float)A->nnz / A->num_rows;
    int max_nnz_per_row = 0;
    int min_nnz_per_row = INT_MAX;
    
    for (int i = 0; i < A->num_rows; i++) {
        int row_nnz = A->row_ptrs[i + 1] - A->row_ptrs[i];
        max_nnz_per_row = max(max_nnz_per_row, row_nnz);
        min_nnz_per_row = min(min_nnz_per_row, row_nnz);
    }
    
    float skewness = (float)max_nnz_per_row / (min_nnz_per_row + 1);
    
    // 选择策略
    if (avg_nnz_per_row < 4) {
        config.kernel_type = SCALAR_CSR;  // 短行用 Scalar
    } else if (skewness < 10) {
        config.kernel_type = VECTOR_CSR;  // 均匀分布用 Vector
    } else {
        config.kernel_type = MERGE_PATH;  // 高度不均匀用 Merge Path
    }
    
    return config;
}
```

### 6. PageRank 算法

```cpp
struct PageRankConfig {
    float damping_factor;    // 阻尼系数，通常 0.85
    float tolerance;         // 收敛阈值，默认 1e-6
    int max_iterations;      // 最大迭代次数
};

struct PageRankResult {
    float* ranks;            // 排名分数 [num_nodes]
    int iterations;          // 实际迭代次数
    float final_residual;    // 最终残差
    bool converged;          // 是否收敛
};

// PageRank 迭代: r_{k+1} = d * A * r_k + (1-d) / n
// 其中 A 是列归一化的邻接矩阵
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,  // 邻接矩阵 (CSR 格式)
    const PageRankConfig* config
);

// 处理悬挂节点 (无出边的节点)
void handle_dangling_nodes(CSRMatrix* adj_matrix, float* dangling_mask);
```


## Data Models

### 矩阵存储内存布局

```
CSR Format Memory Layout:
========================
values:      [v0, v1, v2, v3, v4, v5, ...]     // 按行顺序存储非零值
col_indices: [c0, c1, c2, c3, c4, c5, ...]     // 对应的列索引
row_ptrs:    [0, 2, 5, 7, ...]                 // 每行起始位置

Example: 3x4 matrix with 5 non-zeros
| 1 0 2 0 |     values:      [1, 2, 3, 4, 5]
| 0 3 4 0 |  => col_indices: [0, 2, 1, 2, 3]
| 0 0 0 5 |     row_ptrs:    [0, 2, 4, 5]


ELL Format Memory Layout (Column-Major):
========================================
max_nnz_per_row = 2

values (column-major):       col_indices (column-major):
| 1 2 |                      | 0  2 |
| 3 4 |  stored as           | 1  2 |  stored as
| 5 0 |  [1,3,5,2,4,0]       | 3 -1 |  [0,1,3,2,2,-1]

GPU 访问模式: 相邻线程访问连续内存地址 (coalesced)
```

### 错误码定义

```cpp
enum SpMVError {
    SPMV_SUCCESS = 0,
    SPMV_ERROR_INVALID_DIMENSION = -1,
    SPMV_ERROR_CUDA_MALLOC = -2,
    SPMV_ERROR_CUDA_MEMCPY = -3,
    SPMV_ERROR_KERNEL_LAUNCH = -4,
    SPMV_ERROR_INVALID_FORMAT = -5,
    SPMV_ERROR_FILE_IO = -6,
    SPMV_ERROR_OUT_OF_MEMORY = -7
};

const char* spmv_error_string(SpMVError err);
```

### GPU 资源管理 (RAII)

```cpp
// RAII 风格的 GPU 内存管理
template<typename T>
class CudaBuffer {
public:
    CudaBuffer(size_t count) : size_(count) {
        cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
        if (err != cudaSuccess) {
            throw CudaException(err);
        }
    }
    
    ~CudaBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // 禁止拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // 允许移动
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }
    
    T* get() { return ptr_; }
    size_t size() const { return size_; }
    
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: CSR Dense-to-Sparse Round Trip

*For any* dense matrix M, converting M to CSR format and then reconstructing the dense matrix should produce a matrix equivalent to M (all elements match within floating-point tolerance).

**Validates: Requirements 1.2**

### Property 2: CSR Element Lookup Correctness

*For any* sparse matrix in CSR format and any valid position (i, j), the `csr_get_element` function should return the correct value (the non-zero value if present, or 0.0 otherwise).

**Validates: Requirements 1.3**

### Property 3: CSR Serialization Round Trip

*For any* valid CSR matrix, serializing to binary format and deserializing should produce an equivalent CSR structure (same dimensions, same non-zero count, same values and indices).

**Validates: Requirements 1.5**

### Property 4: ELL Dense-to-Sparse Round Trip

*For any* dense matrix M, converting M to ELL format and then reconstructing the dense matrix should produce a matrix equivalent to M.

**Validates: Requirements 2.2**

### Property 5: ELL Padding Correctness

*For any* sparse matrix converted to ELL format, rows with fewer than max_nnz_per_row non-zero elements should have remaining positions filled with 0.0 values and -1 column indices.

**Validates: Requirements 2.3**

### Property 6: ELL Column-Major Layout

*For any* ELL matrix, the element at logical position (row, k) should be stored at memory index (k * num_rows + row), ensuring coalesced access for consecutive threads.

**Validates: Requirements 2.4**

### Property 7: ELL Serialization Round Trip

*For any* valid ELL matrix, serializing to binary format and deserializing should produce an equivalent ELL structure.

**Validates: Requirements 2.5**

### Property 8: SpMV CSR Correctness

*For any* sparse matrix A in CSR format and any input vector x of matching dimension, the GPU SpMV result y_gpu should match the CPU reference result y_cpu within relative tolerance 1e-6 (i.e., |y_gpu[i] - y_cpu[i]| / |y_cpu[i]| < 1e-6 for all i where y_cpu[i] != 0).

**Validates: Requirements 3.1, 3.3**

### Property 9: SpMV ELL Correctness

*For any* sparse matrix A in ELL format and any input vector x of matching dimension, the GPU SpMV result should match the CPU reference result within relative tolerance 1e-6.

**Validates: Requirements 3.2, 3.3**

### Property 10: SpMV Dimension Validation

*For any* sparse matrix A and input vector x where x.length != A.num_cols, the SpMV function should return error code SPMV_ERROR_INVALID_DIMENSION without performing computation.

**Validates: Requirements 3.5, 8.5**

### Property 11: Kernel Selector Validity

*For any* valid CSR matrix, the kernel selector function should return a valid SpMVConfig with a supported kernel type and reasonable block size (32 <= block_size <= 1024, power of 2).

**Validates: Requirements 4.5**

### Property 12: Bandwidth Metrics Validity

*For any* completed SpMV operation, the returned bandwidth metrics should have: achieved_bandwidth >= 0, efficiency in range [0, 1], and theoretical_bandwidth > 0.

**Validates: Requirements 5.5**

### Property 13: Benchmark Metrics Completeness

*For any* benchmark run, the result should contain valid execution_time (> 0), gflops (>= 0), bandwidth_gb_s (>= 0), and statistical measures (avg, min, max, stddev) where min <= avg <= max.

**Validates: Requirements 6.1, 6.3**

### Property 14: Benchmark JSON Round Trip

*For any* benchmark result, serializing to JSON and parsing back should produce equivalent data.

**Validates: Requirements 6.5**

### Property 15: PageRank Score Invariants

*For any* valid adjacency matrix representing a graph, the PageRank algorithm should produce scores where: (1) all scores are non-negative, (2) scores sum to 1.0 within tolerance, (3) the algorithm converges (final residual < tolerance) or reaches max iterations.

**Validates: Requirements 7.1, 7.2**

### Property 16: PageRank Top-K Ordering

*For any* PageRank result and k > 0, the top-k nodes returned should be sorted in descending order by rank score, and no node outside top-k should have a higher score than any node in top-k.

**Validates: Requirements 7.5**


## Error Handling

### 错误处理策略

1. **输入验证**：所有公共 API 在执行前验证参数
   - 矩阵维度必须为正数
   - 向量维度必须匹配矩阵
   - 指针不能为 NULL

2. **CUDA 错误检查**：每个 CUDA API 调用后检查错误
   ```cpp
   #define CUDA_CHECK(call) do { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           return spmv_cuda_error(err); \
       } \
   } while(0)
   ```

3. **资源清理**：使用 RAII 确保异常安全
   - GPU 内存通过 CudaBuffer 管理
   - 失败时自动释放已分配资源

4. **错误传播**：错误码通过返回值传播，详细信息通过日志记录

### 边界情况处理

| 情况 | 处理方式 |
|------|----------|
| 空矩阵 (0 行或 0 列) | 返回空结果向量 |
| 全零行 | SpMV 正常处理，结果为 0 |
| 单元素矩阵 | 正常处理 |
| 极大矩阵 (超出 GPU 内存) | 返回 SPMV_ERROR_OUT_OF_MEMORY |
| NaN/Inf 输入值 | 传播到输出 (IEEE 754 语义) |

## Testing Strategy

### 测试框架

- **单元测试**: Google Test (gtest)
- **属性测试**: 自定义生成器 + gtest
- **性能测试**: CUDA Events 计时

### 测试类型

#### 1. 单元测试
- 测试特定示例和边界情况
- 验证错误处理路径
- 测试格式转换的具体案例

#### 2. 属性测试 (Property-Based Testing)
- 使用随机生成的稀疏矩阵验证通用属性
- 每个属性测试运行至少 100 次迭代
- 生成器策略：
  - 矩阵大小: 1x1 到 10000x10000
  - 稀疏度: 0.001 到 0.5
  - 行长度分布: 均匀、偏斜、极端

#### 3. 集成测试
- 端到端 SpMV 流程测试
- PageRank 收敛性测试
- 多 Kernel 一致性测试

### 测试矩阵生成器

```cpp
// 随机稀疏矩阵生成器
struct SparseMatrixGenerator {
    int min_rows = 1, max_rows = 1000;
    int min_cols = 1, max_cols = 1000;
    float min_density = 0.001, max_density = 0.3;
    
    enum RowDistribution {
        UNIFORM,        // 每行非零元素数量相近
        POWER_LAW,      // 幂律分布 (模拟真实图)
        EXTREME_SKEW    // 极端不均匀
    };
    
    CSRMatrix* generate(RowDistribution dist = UNIFORM);
};

// 测试向量生成器
float* generate_random_vector(int size, float min_val = -1.0, float max_val = 1.0);
```

### 属性测试标注格式

每个属性测试必须包含注释引用设计文档中的属性：

```cpp
// **Feature: spmv-gpu, Property 8: SpMV CSR Correctness**
// **Validates: Requirements 3.1, 3.3**
TEST(SpMVPropertyTest, CSRCorrectnessProperty) {
    for (int iter = 0; iter < 100; iter++) {
        auto matrix = generator.generate();
        auto x = generate_random_vector(matrix->num_cols);
        // ... test implementation
    }
}
```
