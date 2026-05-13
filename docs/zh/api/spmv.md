# SpMV 计算

核心 SpMV 计算 API，包括 Kernel 类型、配置和执行。

## Kernel 类型

```cpp
enum KernelType {
    SCALAR_CSR,  // 1 线程/行 - 适合极稀疏矩阵
    VECTOR_CSR,  // 1 Warp/行 - 适合均匀分布矩阵
    MERGE_PATH,  // 负载均衡 - 适合倾斜矩阵
    ELL_KERNEL   // ELL 格式专用
};
```

## 配置结构

### SpMVConfig

```cpp
struct SpMVConfig {
    KernelType kernel_type;
    int block_size;    // CUDA 块大小（默认 256）
    bool use_texture;  // 是否使用纹理缓存
};
```

### SpMVThresholds

自动选择 Kernel 的可自定义阈值：

```cpp
struct SpMVThresholds {
    float avg_nnz_threshold;     // 默认 4.0
    float skewness_threshold;    // 默认 10.0
    int texture_cols_threshold;  // 默认 10000
};
```

### SpMVExecutionContext

可选的执行上下文，用于跨迭代复用纹理缓存：

```cpp
struct SpMVExecutionContext {
    cudaTextureObject_t tex_x;  // 纹理对象
    const float* cached_x;      // 缓存的 x 指针
    size_t cached_x_length;     // 缓存的 x 长度
    bool texture_enabled;       // 是否启用纹理

    SpMVExecutionContext();
    ~SpMVExecutionContext();

    void reset();  // 重置并释放纹理对象

    // 禁用拷贝，允许移动
    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext(SpMVExecutionContext&&) noexcept;
};
```

### SpMVResult

```cpp
struct SpMVResult {
    float* y;              // 输出向量（设备指针）
    float elapsed_ms;      // 执行时间（毫秒）
    float gflops;          // 计算性能 (GFLOPS)
    float bandwidth_gb_s;  // 内存带宽 (GB/s)
    int error_code;        // 0 = 成功，负数 = 错误
};
```

## 核心函数

### 自动配置

```cpp
// 自动选择最优 Kernel 配置
SpMVConfig spmv_auto_config(const CSRMatrix* A);
```

### CSR SpMV

```cpp
// GPU CSR 格式 SpMV
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 表示自动检测
                    SpMVExecutionContext* context = nullptr);
```

### ELL SpMV

```cpp
// GPU ELL 格式 SpMV
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);
```

### CPU 参考实现

```cpp
// CPU 参考实现（用于验证）
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);
```

### 阈值管理

```cpp
SpMVThresholds spmv_get_thresholds();
void spmv_set_thresholds(const SpMVThresholds& thresholds);
```

## 错误处理

```cpp
enum class SpMVError {
    SUCCESS = 0,             // 成功
    INVALID_DIMENSION = -1,  // 维度不匹配
    CUDA_MALLOC = -2,        // GPU 内存分配失败
    CUDA_MEMCPY = -3,        // 内存拷贝失败
    KERNEL_LAUNCH = -4,      // Kernel 启动失败
    INVALID_FORMAT = -5,     // 无效格式
    FILE_IO = -6,            // 文件 IO 错误
    OUT_OF_MEMORY = -7,      // 内存不足
    INVALID_ARGUMENT = -8    // 无效参数
};

const char* spmv_error_string(SpMVError err);  // 获取错误描述字符串
```

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
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config);

    // 4. 检查结果
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

## 头文件

```cpp
#include <spmv/benchmark.h>    // 性能测试
#include <spmv/csr_matrix.h>   // CSR 矩阵
#include <spmv/cuda_buffer.h>  // RAII 内存管理
#include <spmv/ell_matrix.h>   // ELL 矩阵
#include <spmv/pagerank.h>     // PageRank
#include <spmv/spmv.h>         // 主接口 + SpMV 计算
```
