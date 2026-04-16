---
layout: default
title: API 参考
lang: zh
---

<p align="right">
  <a href="api.en.html">🇺🇸 English</a>
</p>

# 📚 API 参考

本文档提供 GPU SpMV 库的完整 API 参考，包含所有公共接口、数据结构和错误码。

---

## 目录

- [头文件概览](#头文件概览)
- [稀疏矩阵格式](#稀疏矩阵格式)
  - [CSR 格式](#csr-compressed-sparse-row)
  - [ELL 格式](#ell-ellpack)
- [SpMV 接口](#spmv-接口)
- [RAII 工具类](#raii-工具类)
- [基准测试](#基准测试)
- [PageRank 算法](#pagerank-算法)
- [错误处理](#错误处理)

---

## 头文件概览

| 头文件 | 功能描述 | 主要组件 |
|:-------|:---------|:---------|
| `<spmv/common.h>` | 错误码与基础定义 | `SpMVError`, `CUDA_CHECK_*` |
| `<spmv/cuda_buffer.h>` | RAII GPU 内存管理 | `CudaBuffer<T>` |
| `<spmv/csr_matrix.h>` | CSR 稀疏矩阵 | `CSRMatrix`, `csr_*` 函数 |
| `<spmv/ell_matrix.h>` | ELL 稀疏矩阵 | `ELLMatrix`, `ell_*` 函数 |
| `<spmv/spmv.h>` | SpMV 计算接口 | `spmv_csr`, `spmv_ell` |
| `<spmv/bandwidth.h>` | 带宽度量 | `BandwidthMetrics` |
| `<spmv/benchmark.h>` | 性能测试 | `benchmark_csr`, `benchmark_ell` |
| `<spmv/pagerank.h>` | PageRank 算法 | `pagerank`, `PageRankConfig` |

---

## 稀疏矩阵格式

### CSR (Compressed Sparse Row)

**头文件**: `<spmv/csr_matrix.h>`

#### 数据结构

```cpp
struct CSRMatrix {
    int num_rows;          // 行数
    int num_cols;          // 列数
    int nnz;              // 非零元素总数

    float* values;         // 非零值 [nnz]
    int* col_indices;      // 列索引 [nnz]
    int* row_ptrs;         // 行指针 [num_rows + 1]

    // GPU 端指针
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

#### 核心函数

| 函数 | 签名 | 说明 |
|:-----|:-----|:-----|
| `csr_create` | `CSRMatrix* csr_create(int rows, int cols, int nnz)` | 创建 CSR 矩阵 |
| `csr_destroy` | `void csr_destroy(CSRMatrix* csr)` | 销毁 CSR 矩阵 |
| `csr_from_dense` | `int csr_from_dense(CSRMatrix* csr, const float* data, int rows, int cols)` | 从稠密矩阵转换 |
| `csr_to_gpu` | `int csr_to_gpu(CSRMatrix* csr)` | 上传至 GPU |
| `csr_from_gpu` | `int csr_from_gpu(CSRMatrix* csr)` | 从 GPU 下载 |
| `csr_compute_stats` | `CSRStats csr_compute_stats(const CSRMatrix* csr)` | 计算统计信息 |

**统计信息结构**:
```cpp
struct CSRStats {
    float avg_nnz_per_row;   // 平均每行非零元素
    int max_nnz_per_row;     // 最大行非零元素
    int min_nnz_per_row;     // 最小行非零元素
    float skewness;          // 偏斜度
};
```

---

### ELL (ELLPACK)

**头文件**: `<spmv/ell_matrix.h>`

#### 数据结构

```cpp
struct ELLMatrix {
    int num_rows;
    int num_cols;
    int max_nnz_per_row;   // 每行最大非零数
    int nnz;              // 实际非零数

    float* values;         // Column-major 存储
    int* col_indices;      // 列索引 (-1 表示填充)

    float* d_values;
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

#### 核心函数

| 函数 | 签名 | 说明 |
|:-----|:-----|:-----|
| `ell_create` | `ELLMatrix* ell_create(int rows, int cols, int max_nnz)` | 创建 ELL 矩阵 |
| `ell_destroy` | `void ell_destroy(ELLMatrix* ell)` | 销毁 ELL 矩阵 |
| `ell_from_csr` | `int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr)` | CSR 转 ELL |
| `ell_to_gpu` | `int ell_to_gpu(ELLMatrix* ell)` | 上传至 GPU |

---

## SpMV 接口

**头文件**: `<spmv/spmv.h>`

### 配置常量

```cpp
namespace spmv {
    constexpr int WARP_SIZE = 32;                    // Warp 大小
    constexpr int MIN_BLOCK_SIZE = 32;               // 最小 Block
    constexpr int MAX_BLOCK_SIZE = 1024;             // 最大 Block
    constexpr int DEFAULT_BLOCK_SIZE = 256;          // 默认 Block
    constexpr int TEXTURE_CACHE_THRESHOLD_COLS = 10000;  // 纹理缓存阈值
}
```

### 配置结构

```cpp
struct SpMVConfig {
    enum KernelType {
        SCALAR_CSR,   // 一个线程/行
        VECTOR_CSR,   // 一个 Warp/行
        MERGE_PATH,   // 工作量均匀划分
        ELL_KERNEL    // ELL 专用
    };

    KernelType kernel_type;
    int block_size = DEFAULT_BLOCK_SIZE;
    bool use_texture = false;
};
```

### Kernel 选择策略

| KernelType | 适用条件 | 说明 |
|:-----------|:---------|:-----|
| `SCALAR_CSR` | `avg_nnz_per_row < 4` | 极稀疏，最小线程开销 |
| `VECTOR_CSR` | `skewness < 10` | 均匀分布，Warp 协作 |
| `MERGE_PATH` | `skewness >= 10` | 不均匀，完美负载均衡 |
| `ELL_KERNEL` | ELL 格式专用 | 完全合并访存 |

### 核心函数

```cpp
// 自动选择最优配置
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* A,
                  const float* d_x,
                  float* d_y,
                  const SpMVConfig* config = nullptr,
                  int vec_size = -1,
                  SpMVExecutionContext* context = nullptr);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* A,
                  const float* d_x,
                  float* d_y,
                  const SpMVConfig* config = nullptr,
                  int vec_size = -1,
                  SpMVExecutionContext* context = nullptr);
```

### 结果结构

```cpp
struct SpMVResult {
    float* y;              // 输出向量指针
    float elapsed_ms;      // 执行时间 (ms)
    float gflops;          // 计算吞吐量
    float bandwidth_gb_s;  // 内存带宽
    int error_code;        // 错误码 (0=成功)
};
```

### 执行上下文

```cpp
// 用于复用纹理对象
class SpMVExecutionContext {
public:
    SpMVExecutionContext();
    ~SpMVExecutionContext();
    void reset();
};
```

**使用示例（多次调用复用）**:
```cpp
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

for (int i = 0; i < 100; i++) {
    // 纹理对象只创建一次，后续复用
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols, &context);
}
```

---

## RAII 工具类

### CudaBuffer

**头文件**: `<spmv/cuda_buffer.h>`

自动管理 GPU 内存的 RAII 模板类：

```cpp
template<typename T>
class CudaBuffer {
public:
    // 构造/析构
    explicit CudaBuffer(size_t n);
    ~CudaBuffer();

    // 数据传输
    void copyFromHost(const T* host_ptr, size_t count);
    void copyToHost(T* host_ptr, size_t count) const;

    // 初始化
    void memset(int value);
    void fill(T value);

    // 访问
    T* get();
    const T* get() const;
    size_t size() const;
    size_t bytes() const;
    bool empty() const;

    // 调整大小
    void resize(size_t new_size);
    void release();

    // 移动语义 (禁用拷贝)
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
};
```

**示例用法**:
```cpp
// 创建缓冲区
CudaBuffer<float> d_x(1000);
CudaBuffer<float> d_y(1000);

// 初始化
d_x.memset(0);

// 主机 → 设备
d_x.copyFromHost(host_data, 1000);

// 设备 → 主机
d_y.copyToHost(result_data, 1000);
```

---

## 基准测试

**头文件**: `<spmv/benchmark.h>`

### 配置结构

```cpp
struct BenchmarkConfig {
    int num_warmup_runs = 5;    // 预热次数
    int num_runs = 20;          // 测试次数
    bool compare_cpu = true;    // 是否对比 CPU
};
```

### 结果结构

```cpp
struct BenchmarkResult {
    std::string name;
    float execution_time_ms;
    float gflops;
    float bandwidth_gb_s;

    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    float stddev_time_ms;
    int num_runs;
    int error_code;
};

struct ComparisonResult {
    BenchmarkResult gpu_result;
    BenchmarkResult cpu_result;
    float speedup;
    int error_code;
};
```

### 测试函数

```cpp
// CSR 基准测试
BenchmarkResult benchmark_csr(const CSRMatrix* A,
                             const float* x,
                             const SpMVConfig* config = nullptr,
                             const BenchmarkConfig* bench_config = nullptr);

// ELL 基准测试
BenchmarkResult benchmark_ell(const ELLMatrix* A,
                             const float* x,
                             const BenchmarkConfig* bench_config = nullptr);

// GPU vs CPU 对比
ComparisonResult compare_gpu_cpu_csr(const CSRMatrix* A,
                                     const float* x,
                                     const SpMVConfig* config = nullptr,
                                     const BenchmarkConfig* bench_config = nullptr);
```

---

## PageRank 算法

**头文件**: `<spmv/pagerank.h>`

### 配置结构

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // 阻尼系数
    float tolerance = 1e-6f;        // 收敛阈值
    int max_iterations = 100;       // 最大迭代次数
};
```

### 结果结构

```cpp
struct PageRankResult {
    float* ranks;           // 排名分数数组
    int iterations;         // 实际迭代次数
    float final_residual;   // 最终残差
    bool converged;         // 是否收敛
    int error_code;         // 错误码
};

struct TopKNode {
    int node_id;
    float rank;
};
```

### 核心函数

```cpp
// 执行 PageRank
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config = nullptr);

// 释放结果
void pagerank_free(PageRankResult* result);

// 获取 Top-K 节点
void pagerank_top_k(const PageRankResult* result,
                   int num_nodes,
                   int k,
                   TopKNode* top_k);
```

**示例**:
```cpp
// 创建邻接矩阵并归一化
CSRMatrix* adj = /* ... */;
csr_to_gpu(adj);

// 配置
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

// 执行
PageRankResult result = pagerank(adj, &config);

if (result.converged) {
    printf("Converged after %d iterations\n", result.iterations);

    // 获取 Top-10
    std::vector<TopKNode> top10(10);
    pagerank_top_k(&result, adj->num_rows, 10, top10.data());
}

pagerank_free(&result);
csr_destroy(adj);
```

---

## 错误处理

**头文件**: `<spmv/common.h>`

### 错误码枚举

```cpp
enum class SpMVError : int {
    SUCCESS = 0,              // 成功
    INVALID_DIMENSION = -1,   // 维度不匹配
    CUDA_MALLOC = -2,         // GPU 内存分配失败
    CUDA_MEMCPY = -3,         // 内存拷贝失败
    KERNEL_LAUNCH = -4,       // Kernel 启动/执行失败
    INVALID_FORMAT = -5,      // 矩阵格式错误
    FILE_IO = -6,             // 文件 IO 失败
    OUT_OF_MEMORY = -7,       // 内存不足
    INVALID_ARGUMENT = -8     // 参数错误
};
```

### 错误处理辅助

```cpp
// 获取错误描述
const char* spmv_error_string(SpMVError err);

// 检查宏 (返回错误码)
#define CUDA_CHECK_MALLOC(call)
#define CUDA_CHECK_MEMCPY(call)

// 检查宏 (抛出异常，用于 RAII 类)
#define CUDA_CHECK_THROW(call)
```

### 最佳实践

```cpp
// 检查返回的错误码
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);

if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
    fprintf(stderr, "SpMV failed: %s\n",
            spmv_error_string(static_cast<SpMVError>(result.error_code)));
    // 错误处理...
}
```

---

<div align="center">

**[← 架构设计](architecture)** · **[ 示例代码 →](examples)**

</div>
