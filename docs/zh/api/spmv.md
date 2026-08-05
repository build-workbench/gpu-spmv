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
    int block_size;     // CUDA 块大小（默认 256）
    bool enable_timing; // 默认 true
};
```

`enable_timing = true`（默认）时，`spmv_csr`/`spmv_ell` 会阻塞直到 `d_y`
完成，并填充 `SpMVResult` 的计时字段。设为 `false` 时只把 kernel 入队到
stream——不创建 CUDA event、不同步——便于流水线编排；读取 `d_y` 前需自行
同步 stream。异步模式下计时字段保持为 0，且只报告同步的启动失败。

### SpMVThresholds

自动选择 Kernel 的可自定义阈值：

```cpp
struct SpMVThresholds {
    float avg_nnz_threshold;  // 默认 4.0
    float skewness_threshold; // 默认 10.0
};
```

### SpMVResult

```cpp
struct SpMVResult {
    float* y;              // 输出向量（设备指针）
    float elapsed_ms;      // 执行时间（毫秒）
    float gflops;          // 计算性能 (GFLOPS)
    float bandwidth_gb_s;  // 内存带宽 (GB/s)，由 elapsed_ms 推导
    int error_code;        // 0 = 成功，负数 = 错误
};

// error_code 的类型化访问器
SpMVError spmv_result_error(const SpMVResult& result);
```

## 核心函数

### 自动配置

```cpp
// 根据矩阵统计信息自动选择 CSR Kernel 配置
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// ELL 矩阵的配置（单一 Kernel）
SpMVConfig spmv_auto_config_ell(const ELLMatrix* A);
```

### CSR SpMV

```cpp
// GPU CSR 格式 SpMV
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 表示自动检测
                    cudaStream_t stream = nullptr);
```

### ELL SpMV

```cpp
// GPU ELL 格式 SpMV
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    cudaStream_t stream = nullptr);
```

### CPU 参考实现

```cpp
// CPU 参考实现（用于验证）。
// 成功返回 0，输入非法时返回负的错误码。
int spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
int spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);
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
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config);

    // 4. 检查结果
    if (spmv_result_error(result) != SpMVError::SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_error_string(spmv_result_error(result)));
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
#include <spmv/csr_matrix.h>   // CSR 矩阵
#include <spmv/cuda_buffer.h>  // RAII 内存管理
#include <spmv/ell_matrix.h>   // ELL 矩阵
#include <spmv/market_io.h>    // Matrix Market 文件读取器
#include <spmv/spmv.h>         // 主接口 + SpMV 计算
```
