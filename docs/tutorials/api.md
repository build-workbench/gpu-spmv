---
layout: default
title: API 参考
nav_order: 3
has_children: false
lang: zh
---

# 📚 API 参考
{: .no_toc }

完整的 GPU SpMV 公共 API 接口文档。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 头文件概览

| 头文件 | 功能 |
|:-------|:-----|
| `<spmv/common.h>` | 错误码与基础定义 |
| `<spmv/cuda_buffer.h>` | RAII GPU 内存管理 |
| `<spmv/csr_matrix.h>` | CSR 稀疏矩阵 |
| `<spmv/ell_matrix.h>` | ELL 稀疏矩阵 |
| `<spmv/spmv.h>` | SpMV 计算接口 |
| `<spmv/benchmark.h>` | 性能测试框架 |
| `<spmv/pagerank.h>` | PageRank 算法 |

---

## 错误处理

### SpMVError 枚举

```cpp
enum class SpMVError {
    SUCCESS = 0,              // 操作成功
    INVALID_DIMENSION = -1,   // 矩阵或向量维度不匹配
    CUDA_MALLOC = -2,         // GPU 内存分配失败
    CUDA_MEMCPY = -3,         // GPU 内存拷贝失败
    KERNEL_LAUNCH = -4,       // CUDA kernel 启动失败
    INVALID_FORMAT = -5,      // 无效的稀疏矩阵格式
    FILE_IO = -6,             // 文件读写错误
    OUT_OF_MEMORY = -7,       // 主机/设备内存不足
    INVALID_ARGUMENT = -8     // 无效参数
};
```

### 错误字符串

```cpp
const char* spmv_error_string(SpMVError err);
```

返回描述错误的可读字符串。

**示例**:
```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
if (result.error != SpMVError::SUCCESS) {
    fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
}
```

---

## 数据结构

### CSRMatrix

CSR (Compressed Sparse Row) 格式稀疏矩阵。

```cpp
struct CSRMatrix {
    int num_rows;           // 行数
    int num_cols;           // 列数
    int nnz;                // 非零元素总数

    float* values;          // 非零值数组 [nnz]
    int* col_indices;       // 列索引数组 [nnz]
    int* row_ptrs;          // 行指针数组 [num_rows + 1]

    // GPU 设备指针
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

**不变式**:
- `row_ptrs[0] == 0`
- `row_ptrs[num_rows] == nnz`
- `row_ptrs[i] <= row_ptrs[i+1]` (对所有 i)

### CSRMatrix API

#### 创建矩阵

```cpp
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
```

创建空的 CSR 矩阵结构。

**参数**:
- `num_rows`: 矩阵行数
- `num_cols`: 矩阵列数
- `nnz`: 非零元素数量

**返回**: 新分配的 CSRMatrix 指针

#### 从密集矩阵转换

```cpp
void csr_from_dense(CSRMatrix* csr, const float* dense, 
                    int num_rows, int num_cols);
```

将密集矩阵转换为 CSR 格式。

**参数**:
- `csr`: 目标 CSR 矩阵
- `dense`: 源密集矩阵 (行优先)
- `num_rows`: 行数
- `num_cols`: 列数

#### 传输到 GPU

```cpp
void csr_to_gpu(CSRMatrix* csr);
```

将 CSR 矩阵数据传输到 GPU。

**参数**:
- `csr`: 要传输的 CSR 矩阵

#### 销毁矩阵

```cpp
void csr_destroy(CSRMatrix* csr);
```

释放 CSR 矩阵占用的所有内存。

**参数**:
- `csr`: 要销毁的 CSR 矩阵

---

### ELLMatrix

ELLPACK 格式稀疏矩阵。

```cpp
struct ELLMatrix {
    int num_rows;           // 行数
    int num_cols;           // 列数
    int max_nnz_per_row;    // 每行最大非零元素数
    int nnz;                // 非零元素总数

    float* values;          // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;       // 列索引数组 [num_rows * max_nnz_per_row]

    // GPU 设备指针
    float* d_values;
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

### ELLMatrix API

#### 创建矩阵

```cpp
ELLMatrix* ell_create(int num_rows, int num_cols, int max_nnz_per_row);
```

创建空的 ELL 矩阵结构。

#### 从 CSR 转换

```cpp
void ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
```

将 CSR 矩阵转换为 ELL 格式。

#### 传输到 GPU

```cpp
void ell_to_gpu(ELLMatrix* ell);
```

将 ELL 矩阵数据传输到 GPU。

#### 销毁矩阵

```cpp
void ell_destroy(ELLMatrix* ell);
```

释放 ELL 矩阵占用的所有内存。

---

## SpMV 计算接口

### SpMVConfig

SpMV 计算配置。

```cpp
struct SpMVConfig {
    KernelType kernel_type;  // Kernel 类型
    bool auto_select;        // 自动选择 kernel
};
```

### KernelType 枚举

```cpp
enum class KernelType {
    SCALAR_CSR,      // Scalar CSR kernel
    VECTOR_CSR,      // Vector CSR kernel
    MERGE_PATH,      // Merge Path kernel
    ELL              // ELL kernel
};
```

### 自动配置

```cpp
SpMVConfig spmv_auto_config(const CSRMatrix* csr);
```

根据矩阵特征自动选择最优 Kernel。

**参数**:
- `csr`: CSR 矩阵

**返回**: 优化的 SpMVConfig

### 执行 SpMV (CSR)

```cpp
SpMVResult spmv_csr(const CSRMatrix* csr, 
                    const float* d_x, 
                    float* d_y, 
                    const SpMVConfig* config, 
                    int n);
```

执行 CSR 格式 SpMV 计算: `y = A * x`

**参数**:
- `csr`: CSR 矩阵 (必须在 GPU 上)
- `d_x`: 输入向量 (GPU 指针)
- `d_y`: 输出向量 (GPU 指针)
- `config`: SpMV 配置
- `n`: 向量维度

**返回**: SpMVResult 包含执行时间和错误码

### 执行 SpMV (ELL)

```cpp
SpMVResult spmv_ell(const ELLMatrix* ell, 
                    const float* d_x, 
                    float* d_y, 
                    int n);
```

执行 ELL 格式 SpMV 计算: `y = A * x`

**参数**:
- `ell`: ELL 矩阵 (必须在 GPU 上)
- `d_x`: 输入向量 (GPU 指针)
- `d_y`: 输出向量 (GPU 指针)
- `n`: 向量维度

**返回**: SpMVResult 包含执行时间和错误码

### SpMVResult

```cpp
struct SpMVResult {
    SpMVError error;      // 错误码
    float time_ms;        // 执行时间 (毫秒)
};
```

---

## RAII 内存管理

### CudaBuffer<T>

模板类，用于自动管理 GPU 内存。

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();
    
    T* data();
    const T* data() const;
    size_t size() const;
    
    // 禁止拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // 允许移动
    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
};
```

**示例**:
```cpp
{
    CudaBuffer<float> buffer(1000);
    // 使用 buffer.data() 访问 GPU 指针
    // 离开作用域时自动释放
}
```

---

## 性能基准

### BenchmarkConfig

```cpp
struct BenchmarkConfig {
    int iterations;           // 迭代次数
    bool warmup;              // 是否预热
    bool print_details;       // 是否打印详细信息
};
```

### 运行基准测试

```cpp
void spmv_benchmark(const CSRMatrix* csr, 
                    const BenchmarkConfig* config);
```

运行 SpMV 基准测试。

**参数**:
- `csr`: CSR 矩阵
- `config`: 基准测试配置

---

## PageRank 算法

### PageRankConfig

```cpp
struct PageRankConfig {
    float damping;            // 阻尼系数 (默认 0.85)
    float tolerance;          // 收敛阈值 (默认 1e-6)
    int max_iterations;       // 最大迭代次数 (默认 100)
};
```

### 执行 PageRank

```cpp
SpMVResult spmv_pagerank(const CSRMatrix* csr, 
                         float* d_rank, 
                         const PageRankConfig* config);
```

执行 PageRank 计算。

**参数**:
- `csr`: CSR 矩阵 (邻接矩阵)
- `d_rank`: 排名向量 (GPU 指针)
- `config`: PageRank 配置

**返回**: SpMVResult 包含收敛信息

---

## 最佳实践

### 1. 资源管理

使用 RAII 模式自动管理资源:

```cpp
// ✅ 推荐
void process() {
    CudaBuffer<float> x(1000);
    CudaBuffer<float> y(1000);
    // 自动清理
}

// ❌ 避免
void process() {
    float* x;
    cudaMalloc(&x, 1000 * sizeof(float));
    // 容易忘记 cudaFree
}
```

### 2. 错误处理

始终检查返回的错误码:

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
if (result.error != SpMVError::SUCCESS) {
    // 处理错误
    fprintf(stderr, "SpMV failed: %s\n", 
            spmv_error_string(result.error));
    return result.error;
}
```

### 3. 性能优化

复用执行上下文以获取最佳性能:

```cpp
SpMVExecutionContext ctx;
for (int i = 0; i < 100; i++) {
    // 纹理对象只创建一次，后续复用
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
}
```

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>完整 API 规范见 <a href="https://github.com/LessUp/gpu-spmv/blob/main/specs/api/public-api.md">specs/api/public-api.md</a></p>
</div>
