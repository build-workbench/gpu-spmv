---
layout: default
title: API 参考
---

# API 参考

本文档提供 GPU SpMV 库的完整 API 参考。

---

## 目录

- [头文件概览](#头文件概览)
- [稀疏矩阵格式](#稀疏矩阵格式)
  - [CSR 格式](#csr-compressed-sparse-row)
  - [ELL 格式](#ell-ellpack)
- [SpMV 接口](#spmv-接口)
  - [配置常量](#配置常量)
  - [核心函数](#核心函数)
  - [执行上下文](#执行上下文)
- [RAII 工具类](#raii-工具类)
- [基准测试](#基准测试)
- [带宽度量](#带宽度量)
- [PageRank 算法](#pagerank-算法)
- [错误处理](#错误处理)

---

## 头文件概览

| 头文件 | 功能 |
|--------|------|
| `spmv/common.h` | 错误码、CUDA 检查宏、异常类 |
| `spmv/cuda_buffer.h` | RAII GPU 内存管理 |
| `spmv/csr_matrix.h` | CSR 稀疏矩阵格式 |
| `spmv/ell_matrix.h` | ELL 稀疏矩阵格式 |
| `spmv/spmv.h` | SpMV 接口、Kernel 选择 |
| `spmv/bandwidth.h` | 带宽度量计算 |
| `spmv/benchmark.h` | 基准测试框架 |
| `spmv/pagerank.h` | PageRank 图算法 |

---

## 稀疏矩阵格式

### CSR (Compressed Sparse Row)

CSR 格式使用三个一维数组存储稀疏矩阵，适合大多数稀疏矩阵场景。

```cpp
#include "spmv/csr_matrix.h"
```

#### 数据结构

```cpp
struct CSRMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int nnz;                // 非零元素总数

    float* values;          // 非零元素值 [nnz]
    int* col_indices;       // 列索引 [nnz]
    int* row_ptrs;          // 行指针 [num_rows + 1]

    // GPU 端指针（调用 csr_to_gpu 后有效）
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;    // 是否拥有主机内存
    bool owns_device_memory;  // 是否拥有设备内存
};
```

#### 内存布局示例

```
稀疏矩阵:               CSR 存储:
| 1 0 2 0 |            values:      [1, 2, 3, 4, 5]
| 0 3 4 0 |     =>     col_indices: [0, 2, 1, 2, 3]
| 0 0 0 5 |            row_ptrs:    [0, 2, 4, 5]
                       (第 0 行: 索引 0-1, 共 2 个元素)
                       (第 1 行: 索引 2-3, 共 2 个元素)
                       (第 2 行: 索引 4,   共 1 个元素)
```

#### 函数接口

| 函数 | 说明 |
|------|------|
| `csr_create(rows, cols, nnz)` | 创建 CSR 矩阵，返回指针 |
| `csr_destroy(csr)` | 释放 CSR 矩阵及所有关联资源 |
| `csr_from_dense(csr, data, rows, cols)` | 从行优先稠密矩阵填充 CSR 数据 |
| `csr_to_dense(csr, dense)` | 将 CSR 转换为稠密矩阵 |
| `csr_get_element(csr, row, col)` | 查询指定位置元素值 |
| `csr_to_gpu(csr)` | 上传 CSR 数据到 GPU |
| `csr_from_gpu(csr)` | 从 GPU 下载 CSR 数据到主机 |
| `csr_free_gpu(csr)` | 仅释放 GPU 端内存 |
| `csr_serialize(csr, filename)` | 序列化到二进制文件 |
| `csr_deserialize(csr, filename)` | 从二进制文件反序列化 |
| `csr_compute_stats(csr)` | 计算矩阵统计信息 |

#### 统计信息结构

```cpp
struct CSRStats {
    float avg_nnz_per_row;   // 平均每行非零元素数
    int max_nnz_per_row;     // 最大每行非零元素数
    int min_nnz_per_row;     // 最小每行非零元素数
    float skewness;          // 偏斜度 = max / (min + 1)
};
```

---

### ELL (ELLPACK)

ELL 格式使用固定宽度的二维数组存储，适合行长度均匀的矩阵，GPU 访存效率最高。

```cpp
#include "spmv/ell_matrix.h"
```

#### 数据结构

```cpp
struct ELLMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int max_nnz_per_row;    // 每行最大非零元素数
    int nnz;                // 实际非零元素总数

    // Column-major 存储以实现 GPU 合并访问
    float* values;          // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;       // 列索引 [-1 表示填充]

    float* d_values;        // GPU 端指针
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

#### Column-Major 存储说明

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

#### 函数接口

| 函数 | 说明 |
|------|------|
| `ell_create(rows, cols, max_nnz)` | 创建 ELL 矩阵 |
| `ell_destroy(ell)` | 释放 ELL 矩阵 |
| `ell_from_dense(ell, data, rows, cols)` | 从稠密矩阵填充 |
| `ell_from_csr(ell, csr)` | 从 CSR 格式转换 |
| `ell_to_dense(ell, dense)` | 转换为稠密矩阵 |
| `ell_get_element(ell, row, col)` | 查询元素值 |
| `ell_to_gpu(ell)` | 上传到 GPU |
| `ell_from_gpu(ell)` | 从 GPU 下载 |
| `ell_free_gpu(ell)` | 释放 GPU 内存 |
| `ell_serialize(ell, filename)` | 序列化到文件 |
| `ell_deserialize(ell, filename)` | 从文件反序列化 |
| `ell_index(row, k, num_rows)` | 计算 Column-major 索引 |

---

## SpMV 接口

```cpp
#include "spmv/spmv.h"
```

### 配置常量

```cpp
namespace spmv {

// CUDA 配置常量
constexpr int WARP_SIZE = 32;                    // Warp 大小
constexpr int MIN_BLOCK_SIZE = 32;               // 最小 Block 大小
constexpr int MAX_BLOCK_SIZE = 1024;             // 最大 Block 大小
constexpr int DEFAULT_BLOCK_SIZE = 256;          // 默认 Block 大小
constexpr int TEXTURE_CACHE_THRESHOLD_COLS = 10000;  // 纹理缓存阈值

}  // namespace spmv
```

### Kernel 类型

```cpp
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,   // 一个线程处理一行
        VECTOR_CSR,   // 一个 Warp 处理一行
        MERGE_PATH,   // 工作量均匀分配
        ELL_KERNEL    // ELL 格式专用
    };

    KernelType kernel_type;  // Kernel 类型
    int block_size;          // CUDA Block 大小
    bool use_texture;        // 是否使用纹理缓存

    // 构造函数
    SpMVConfig();
    SpMVConfig(KernelType type, int block_size, bool use_texture);
};
```

#### Kernel 选择策略

| Kernel | 适用场景 | 优势 |
|--------|----------|------|
| `SCALAR_CSR` | `avg_nnz < 4`，极稀疏矩阵 | 线程开销最小 |
| `VECTOR_CSR` | `skewness < 10`，均匀分布 | Warp 归约高效 |
| `MERGE_PATH` | `skewness >= 10`，高度不均匀 | 完美负载均衡 |
| `ELL_KERNEL` | ELL 格式，行长度接近 | 访存完全合并 |

### 核心函数

```cpp
// CPU 参考实现（用于正确性验证）
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

// 自动选择最优 Kernel 配置
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// GPU SpMV 计算
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

// 维度验证
inline bool spmv_validate_dimensions(int num_cols, int vec_size);
```

### 执行上下文

`SpMVExecutionContext` 用于复用纹理对象，提升多次调用的性能：

```cpp
struct SpMVExecutionContext {
    cudaTextureObject_t tex_x;      // 纹理对象
    const float* cached_x;          // 缓存的输入向量指针
    size_t cached_x_length;         // 缓存的向量长度
    bool texture_enabled;           // 是否启用纹理

    SpMVExecutionContext();
    ~SpMVExecutionContext();        // 自动销毁纹理对象

    void reset();                   // 重置上下文

    // 禁止拷贝，允许移动
    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext(SpMVExecutionContext&&);
};
```

#### 使用示例

```cpp
// 多次调用复用纹理对象
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < iterations; i++) {
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
    // 纹理对象只在第一次调用时创建，后续复用
}
```

### 计算结果

```cpp
struct SpMVResult {
    float* y;              // 输出向量指针
    float elapsed_ms;      // 执行时间（毫秒）
    float gflops;          // 计算吞吐量（GFLOPS）
    float bandwidth_gb_s;  // 实际带宽（GB/s）
    int error_code;        // 错误码，0 表示成功
};
```

---

## RAII 工具类

### CudaBuffer

自动管理 GPU 内存的 RAII 模板类：

```cpp
#include "spmv/cuda_buffer.h"

// 创建缓冲区
CudaBuffer<float> buf(1024);          // 分配 1024 个 float

// 数据传输
buf.copyFromHost(host_ptr, count);    // 主机 → 设备
buf.copyToHost(host_ptr, count);      // 设备 → 主机

// 初始化
buf.memset(0);                        // 填充 0
buf.fill(3.14f);                      // 填充指定值

// 查询
float* ptr = buf.get();               // 获取设备指针
const float* ptr = buf.get();         // const 版本
size_t n = buf.size();                // 元素数量
size_t bytes = buf.bytes();           // 字节数
bool empty = buf.empty();             // 是否为空

// 调整大小
buf.resize(2048);                     // 重新分配

// 释放
buf.release();                        // 手动释放
```

### 内部 RAII 类

以下类供内部使用，不暴露在公共 API 中：

- **CudaTimer**: 基于 CUDA Events 的计时器，用于精确测量 GPU 执行时间
- **ScopedTexture**: 纹理对象生命周期管理，确保异常安全

---

## 基准测试

```cpp
#include "spmv/benchmark.h"
```

### 配置结构

```cpp
struct BenchmarkConfig {
    int num_warmup_runs = 5;   // 预热运行次数
    int num_runs = 20;         // 测试运行次数
    bool compare_cpu = true;   // 是否与 CPU 对比
};
```

### 运行基准测试

```cpp
// CSR SpMV 基准测试
BenchmarkResult benchmark_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config = nullptr,
    const BenchmarkConfig* bench_config = nullptr
);

// ELL SpMV 基准测试
BenchmarkResult benchmark_ell(
    const ELLMatrix* A,
    const float* x,
    const BenchmarkConfig* bench_config = nullptr
);

// GPU vs CPU 对比
ComparisonResult compare_gpu_cpu_csr(
    const CSRMatrix* A,
    const float* x,
    const SpMVConfig* config = nullptr,
    const BenchmarkConfig* bench_config = nullptr
);
```

### 结果结构

```cpp
struct BenchmarkResult {
    std::string name;           // 测试名称
    float execution_time_ms;    // 执行时间
    float gflops;               // GFLOPS
    float bandwidth_gb_s;       // 带宽

    float avg_time_ms;          // 平均时间
    float min_time_ms;          // 最小时间
    float max_time_ms;          // 最大时间
    float stddev_time_ms;       // 时间标准差
    int num_runs;               // 成功运行次数
    int error_code;             // 错误码
};

struct ComparisonResult {
    BenchmarkResult gpu_result;
    BenchmarkResult cpu_result;
    float speedup;              // GPU 加速比
    int error_code;
};
```

### JSON 序列化

```cpp
std::string json = benchmark_to_json(result);
BenchmarkResult parsed = benchmark_from_json(json);
```

---

## 带宽度量

```cpp
#include "spmv/bandwidth.h"
```

### 带宽计算

```cpp
// 计算 CSR SpMV 带宽
BandwidthMetrics compute_bandwidth_csr(
    const CSRMatrix* A,
    float elapsed_ms
);

// 计算 ELL SpMV 带宽
BandwidthMetrics compute_bandwidth_ell(
    const ELLMatrix* A,
    float elapsed_ms
);

// 获取 GPU 峰值带宽
float peak = get_gpu_peak_bandwidth();  // GB/s
```

### 带宽度量结构

```cpp
struct BandwidthMetrics {
    float theoretical_bandwidth_gb_s;  // GPU 理论峰值带宽
    float achieved_bandwidth_gb_s;     // 实际达到的带宽
    float efficiency;                  // 带宽利用率 [0, 1]
};
```

### 带宽计算公式

```
传输字节数 = nnz × (sizeof(float) + sizeof(int))    // values + col_indices
           + (num_rows + 1) × sizeof(int)           // row_ptrs
           + num_cols × sizeof(float)               // x 向量
           + num_rows × sizeof(float)               // y 向量

带宽 = 传输字节数 / 执行时间
```

---

## PageRank 算法

```cpp
#include "spmv/pagerank.h"
```

### 配置

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // 阻尼系数
    float tolerance = 1e-6f;        // 收敛阈值
    int max_iterations = 100;       // 最大迭代次数
};
```

### 运行 PageRank

```cpp
// 执行 PageRank
PageRankResult pagerank(
    const CSRMatrix* adj_matrix,  // 列归一化的邻接矩阵
    const PageRankConfig* config = nullptr
);

// 释放结果
void pagerank_free(PageRankResult* result);

// 获取 Top-K 节点
struct TopKNode {
    int node_id;
    float rank;
};

void pagerank_top_k(
    const PageRankResult* result,
    int num_nodes,
    int k,
    TopKNode* top_k
);
```

### 结果结构

```cpp
struct PageRankResult {
    float* ranks;           // 排名分数数组 [num_nodes]
    int iterations;         // 实际迭代次数
    float final_residual;   // 最终残差
    bool converged;         // 是否收敛
    int error_code;         // 错误码
};
```

### 注意事项

1. 邻接矩阵必须是**列归一化**的（每列和为 1 或 0）
2. 调用前需要上传矩阵到 GPU (`csr_to_gpu`)
3. 结果的 `ranks` 数组需要调用 `pagerank_free` 释放

---

## 错误处理

```cpp
#include "spmv/common.h"
```

### 错误码枚举

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
```

### 错误信息转换

```cpp
const char* spmv_error_string(SpMVError err);
// 返回错误码对应的人类可读描述
```

### CUDA 检查宏

```cpp
// 内存分配检查（返回 CUDA_MALLOC 错误）
CUDA_CHECK_MALLOC(cudaMalloc(&ptr, size));

// 内存拷贝检查（返回 CUDA_MEMCPY 错误）
CUDA_CHECK_MEMCPY(cudaMemcpy(dst, src, size, kind));

// 向后兼容别名
CUDA_CHECK(call)  // 等同于 CUDA_CHECK_MALLOC
```

### 异常类

```cpp
// CudaException 用于 RAII 类中的异常抛出
class CudaException : public std::runtime_error {
public:
    explicit CudaException(cudaError_t err);
    cudaError_t error() const;
};

// 抛出异常的检查宏（用于 CudaBuffer 等类）
CUDA_CHECK_THROW(cudaMalloc(&ptr, size));
```

### 最佳实践

```cpp
// 检查返回的错误码
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);
if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    fprintf(stderr, "SpMV failed: %s\n",
            spmv_error_string(static_cast<SpMVError>(result.error_code)));
    return result.error_code;
}
```
