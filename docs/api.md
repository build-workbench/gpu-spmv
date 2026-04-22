---
layout: default
title: API 参考
nav_order: 3
permalink: /api
lang: zh
---

<p align="right">
  <a href="api.en">🇺🇸 English</a>
</p>

# API 参考
{: .no_toc }

完整的 GPU SpMV 公共 API 接口文档。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 头文件

```cpp
#include <spmv/spmv.h>       // 主接口
#include <spmv/csr_matrix.h> // CSR 矩阵
#include <spmv/ell_matrix.h> // ELL 矩阵
#include <spmv/cuda_buffer.h> // RAII 内存管理
#include <spmv/benchmark.h>  // 性能测试
#include <spmv/pagerank.h>   // PageRank
```

---

## 错误处理

```cpp
enum class SpMVError {
    SUCCESS = 0,              // 成功
    INVALID_DIMENSION = -1,   // 维度不匹配
    CUDA_MALLOC = -2,         // GPU 内存分配失败
    CUDA_MEMCPY = -3,         // 内存拷贝失败
    KERNEL_LAUNCH = -4,       // Kernel 启动失败
    INVALID_FORMAT = -5,      // 无效格式
    FILE_IO = -6,             // 文件 IO 错误
    OUT_OF_MEMORY = -7,       // 内存不足
    INVALID_ARGUMENT = -8     // 无效参数
};

const char* spmv_error_string(SpMVError err);
```

---

## CSR 矩阵

### 数据结构

```cpp
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    float* values;      // [nnz]
    int* col_indices;   // [nnz]
    int* row_ptrs;      // [num_rows + 1]
    float* d_values;    // GPU
    int* d_col_indices; // GPU
    int* d_row_ptrs;    // GPU
};
```

### API

```cpp
// 创建/销毁
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* csr);

// 数据操作
void csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
void csr_to_gpu(CSRMatrix* csr);
void csr_from_gpu(CSRMatrix* csr);
```

---

## ELL 矩阵

```cpp
struct ELLMatrix {
    int num_rows;
    int num_cols;
    int max_nnz_per_row;
    float* values;      // [num_rows * max_nnz_per_row]
    int* col_indices;   // [num_rows * max_nnz_per_row]
    float* d_values;
    int* d_col_indices;
};

// API
ELLMatrix* ell_create(int num_rows, int num_cols, int max_nnz_per_row);
void ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
void ell_to_gpu(ELLMatrix* ell);
void ell_destroy(ELLMatrix* ell);
```

---

## SpMV 计算

### 配置

```cpp
enum class KernelType {
    SCALAR_CSR,    // 单线程/行
    VECTOR_CSR,    // Warp/行
    MERGE_PATH,    // 负载均衡
    ELL            // ELL 格式
};

struct SpMVConfig {
    KernelType kernel_type;
    bool auto_select;
};

struct SpMVResult {
    SpMVError error;
    float time_ms;
    float bandwidth_gbps;
    float bandwidth_utilization;
};
```

### 函数

```cpp
// 自动配置
SpMVConfig spmv_auto_config(const CSRMatrix* csr);

// CSR SpMV
SpMVResult spmv_csr(const CSRMatrix* csr, 
                    const float* d_x, 
                    float* d_y, 
                    const SpMVConfig* config, 
                    int n);

// ELL SpMV
SpMVResult spmv_ell(const ELLMatrix* ell, 
                    const float* d_x, 
                    float* d_y, 
                    int n);
```

---

## RAII 内存管理

```cpp
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();
    
    T* data();
    size_t size() const;
    
    // 禁用拷贝，允许移动
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
};
```

---

## PageRank

```cpp
struct PageRankConfig {
    float damping = 0.85f;
    float tolerance = 1e-6f;
    int max_iterations = 100;
};

SpMVResult pagerank(const CSRMatrix* csr, 
                    float* d_rank, 
                    const PageRankConfig* config);
```

---

## 完整示例

```cpp
#include <spmv/spmv.h>

int main() {
    // 1. 创建 CSR 矩阵
    CSRMatrix* csr = csr_create(1000, 1000, 10000);
    // ... 填充数据 ...
    csr_to_gpu(csr);
    
    // 2. 准备向量
    CudaBuffer<float> d_x(1000), d_y(1000);
    
    // 3. 自动配置并执行
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 1000);
    
    // 4. 检查结果
    if (result.error != SpMVError::SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
        return 1;
    }
    
    printf("Bandwidth: %.1f%%\n", result.bandwidth_utilization * 100);
    
    csr_destroy(csr);
    return 0;
}
```

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>更多示例见 <a href="examples">示例代码</a> 页面</p>
</div>
