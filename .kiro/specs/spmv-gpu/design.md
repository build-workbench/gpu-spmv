# 设计文档：GPU SpMV (稀疏矩阵向量乘法)

> **版本**: v1.0.0
> **状态**: ✅ 已实现
> **最后更新**: 2025-04-16

---

## 概述

本文档描述基于 CUDA 的稀疏矩阵向量乘法 (SpMV) 实现。系统采用分层架构，包含存储层（CSR/ELL 格式）、计算层（多种优化 Kernel）、和应用层（PageRank 算法）。

### 设计目标

| 目标 | 描述 | 验证方式 |
|------|------|----------|
| **正确性** | SpMV 结果与 CPU 参考实现一致 | 属性测试 |
| **性能** | 最大化 GPU 带宽利用率 | 基准测试 |
| **可扩展性** | 支持大规模稀疏矩阵 | 大矩阵测试 |

---

## 系统架构

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

## 组件与接口

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
    
    bool owns_host_memory;
    bool owns_device_memory;
};
```

**内存布局示例**:
```
稀疏矩阵:               CSR 存储:
| 1 0 2 0 |            values:      [1, 2, 3, 4, 5]
| 0 3 4 0 |     =>     col_indices: [0, 2, 1, 2, 3]
| 0 0 0 5 |            row_ptrs:    [0, 2, 4, 5]
                       (第 0 行: 索引 0-1, 共 2 个元素)
                       (第 1 行: 索引 2-3, 共 2 个元素)
                       (第 2 行: 索引 4,   共 1 个元素)
```

#### ELL (ELLPACK) 格式

```cpp
struct ELLMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int max_nnz_per_row;    // 每行最大非零元素数
    int nnz;                // 实际非零元素总数
    
    // Column-major 存储以实现合并访问
    float* values;          // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;       // 列索引 [-1 表示填充]
    
    float* d_values;        // GPU 端指针
    int* d_col_indices;
    
    bool owns_host_memory;
    bool owns_device_memory;
};
```

**Column-Major 存储说明**:
```
稀疏矩阵 (max_nnz_per_row = 2):
| 1 0 2 |     行 0: [1, 2] 列 [0, 2]
| 3 4 0 | =>  行 1: [3, 4] 列 [0, 1]
| 5 0 0 |     行 2: [5, -] 列 [0, -]

Column-major 存储:
values:     [1, 3, 5, 2, 4, 0]     // 按列存储
col_indices: [0, 0, 0, 2, 1, -1]   // -1 表示填充

GPU 访问: 线程 i 访问 values[k*num_rows + i]，连续地址！
```

---

### 2. SpMV Kernel 接口

```cpp
// 统一的 SpMV 配置
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

// 计算结果
struct SpMVResult {
    float* y;               // 输出向量 (GPU)
    float elapsed_ms;       // 执行时间
    float gflops;           // 计算吞吐量
    float bandwidth_gb_s;   // 带宽利用率
    int error_code;         // 0 = 成功
};

// 核心 SpMV 函数
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

// 自动选择最优 Kernel
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

---

### 3. CUDA Kernel 设计

#### 3.1 Scalar CSR Kernel

**策略**: 一个线程处理一行

**适用场景**: `avg_nnz_per_row < 4`（极稀疏矩阵）

**优势**: 实现简单，无同步开销

**劣势**: 长行导致 Warp 空闲

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

**策略**: 一个 Warp (32线程) 协作处理一行

**适用场景**: `skewness < 10`（行长度均匀分布）

**优势**: Warp 归约高效，合并访问良好

**劣势**: 长行仍有瓶颈

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
        
        // Warp 内线程协作处理一行
        for (int j = row_ptrs[warp_id] + lane_id; 
             j < row_ptrs[warp_id + 1]; j += 32) {
            sum += values[j] * x[col_indices[j]];
        }
        
        // Warp 级归约 (使用 shuffle 指令)
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) y[warp_id] = sum;
    }
}
```

#### 3.3 Merge Path Kernel

**策略**: 将行指针和非零元序列视为两条有序路径，通过二分搜索找到均匀分割点

**适用场景**: `skewness >= 10`（高度不均匀分布）

**优势**: 完美负载均衡

**劣势**: 实现复杂

```cpp
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

**Merge Path 可视化**:
```
行指针序列:   [0, 2, 5, 7, 10]    (4 行，共 10 个非零元)
非零元序列:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Merge Path 视角:
         row:  0   1   2   3   4
              [0]─[2]─[5]─[7]─[10]
               │╲  │╲  │╲  │╲
              [0][1][2][3][4][5][6][7][8][9]  (nz)
               0   1   2   3   4   5   6   7   8   9

均匀划分: 对角线切割，每个线程处理相同数量的工作
```

#### 3.4 ELL Kernel

**策略**: Column-major 存储，天然合并访问

**适用场景**: 行长度接近 `max_nnz_per_row`

**优势**: 访存完全合并，无条件分支

**劣势**: 内存浪费（填充）

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
            int idx = k * num_rows + row;  // Column-major 索引
            int col = col_indices[idx];
            if (col >= 0) {  // -1 表示填充
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
```

---

### 4. Kernel 选择策略

```
                    矩阵特征分析
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
    
    // 计算行长度统计
    CSRStats stats = csr_compute_stats(A);
    
    // 选择策略
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

### 5. 带宽优化

#### 5.1 Column-Major 存储 (ELL 格式)

```
Row-major 访问模式 (差):
线程:   T0      T1      T2
        ↓       ↓       ↓
地址: [row0,k0][row1,k0][row2,k0]  ← 不连续！
       [base+0] [base+max_nnz] [base+2*max_nnz]

Column-major 访问模式 (好):
线程:   T0      T1      T2
        ↓       ↓       ↓
地址: [row0,k0][row1,k0][row2,k0]  ← 连续！
       [base+0] [base+1]   [base+2]
```

#### 5.2 纹理缓存

```cpp
// 使用 SpMVExecutionContext 复用纹理对象
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < iterations; i++) {
    // 纹理对象只在第一次调用时创建
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
// context 析构时自动销毁纹理对象
```

#### 5.3 Warp 级归约

```cpp
// 传统 Shared Memory 归约 (有 bank conflict 风险)
__shared__ float sdata[32];
sdata[lane_id] = sum;
for (int offset = 16; offset > 0; offset /= 2) {
    sdata[lane_id] += sdata[lane_id + offset];  // 可能冲突
}

// Shuffle 归约 (无 bank conflict)
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);  // 完全并行
}
```

---

### 6. PageRank 算法

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // 阻尼系数
    float tolerance = 1e-6f;        // 收敛阈值
    int max_iterations = 100;       // 最大迭代次数
};

struct PageRankResult {
    float* ranks;           // 排名分数数组 [num_nodes]
    int iterations;         // 实际迭代次数
    float final_residual;   // 最终残差
    bool converged;         // 是否收敛
    int error_code;         // 错误码
};

// PageRank 迭代: r_{k+1} = d * A * r_k + (1-d) / n
// 其中 A 是列归一化的邻接矩阵
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,
    const PageRankConfig* config
);
```

---

### 7. 错误处理

#### 错误码定义

```cpp
enum class SpMVError {
    SUCCESS = 0,
    INVALID_DIMENSION = -1,   // 矩阵或向量维度不匹配
    CUDA_MALLOC = -2,         // GPU 内存分配失败
    CUDA_MEMCPY = -3,         // GPU 内存拷贝失败
    KERNEL_LAUNCH = -4,       // CUDA Kernel 启动/执行失败
    INVALID_FORMAT = -5,      // 稀疏矩阵格式非法
    FILE_IO = -6,             // 文件读写失败
    OUT_OF_MEMORY = -7,       // 主机/设备内存不足
    INVALID_ARGUMENT = -8     // 参数非法
};

const char* spmv_error_string(SpMVError err);
```

#### CUDA 检查宏

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

#### RAII 资源管理

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();
    
    // 禁止拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // 允许移动
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

## 正确性属性

### 属性测试列表

| ID | 属性名称 | 验证需求 |
|----|----------|----------|
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

### 属性测试模板

```cpp
// **Property 8: SpMV CSR Correctness**
// **Validates: Requirements 3.1, 3.3**
TEST(SpMVPropertyTest, CSRCorrectnessProperty) {
    for (int iter = 0; iter < 100; iter++) {
        auto matrix = generator.generate();
        auto x = generate_random_vector(matrix->num_cols);
        
        // GPU 计算
        SpMVResult gpu_result = spmv_csr(matrix, d_x, d_y, &config);
        
        // CPU 参考
        spmv_cpu_csr(matrix, x.data(), y_cpu.data());
        
        // 验证相对误差
        for (int i = 0; i < matrix->num_rows; i++) {
            if (y_cpu[i] != 0) {
                EXPECT_LT(abs(y_gpu[i] - y_cpu[i]) / abs(y_cpu[i]), 1e-6);
            }
        }
    }
}
```

---

## 测试策略

### 测试类型

| 类型 | 框架 | 目的 |
|------|------|------|
| 单元测试 | Google Test | 测试特定示例和边界情况 |
| 属性测试 | Google Test + 随机生成 | 验证通用属性 |
| 性能测试 | CUDA Events | 测量执行时间和带宽 |

### 测试矩阵生成器

```cpp
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
```

### 边界情况处理

| 情况 | 处理方式 |
|------|----------|
| 空矩阵 (0 行或 0 列) | 返回空结果向量 |
| 全零行 | SpMV 正常处理，结果为 0 |
| 单元素矩阵 | 正常处理 |
| 极大矩阵 (超出 GPU 内存) | 返回 OUT_OF_MEMORY 错误 |
| NaN/Inf 输入值 | 传播到输出 (IEEE 754 语义) |
