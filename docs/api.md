-- -layout : default title : API 参考 nav_order : 3 permalink : / api lang : zh-- -

    <p align = "right"><a href = "api.en">🇺🇸 English</ a></ p>
#API 参考
    { :.no_toc}

    完整的 GPU SpMV 公共 API 接口文档。 {: .fs-6 .fw-300
}

##目录 {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

   ##头文件

```cpp
#include <spmv/benchmark.h>    // 性能测试
#include <spmv/csr_matrix.h>   // CSR 矩阵
#include <spmv/cuda_buffer.h>  // RAII 内存管理
#include <spmv/ell_matrix.h>   // ELL 矩阵
#include <spmv/pagerank.h>     // PageRank
#include <spmv/spmv.h>         // 主接口 + SpMV 计算
```

   -- -

   ##错误处理

```cpp enum class SpMVError {
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

    -- -

    ##CSR 矩阵

    ## #数据结构

```cpp struct CSRMatrix {
    int num_rows;  // 行数
    int num_cols;  // 列数
    int nnz;       // 非零元素总数

    float* values;     // 非零值数组 [nnz]
    int* col_indices;  // 列索引数组 [nnz]
    int* row_ptrs;     // 行指针数组 [num_rows + 1]

    // GPU 设备指针
    float* d_values;     // GPU 端值数组
    int* d_col_indices;  // GPU 端列索引
    int* d_row_ptrs;     // GPU 端行指针

    // 内存所有权标志
    bool owns_host_memory;    // 是否拥有主机内存
    bool owns_device_memory;  // 是否拥有设备内存
};
```

    ## #核心函数

```cpp
    // 创建与销毁
    CSRMatrix*
    csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* mat);

// 数据转换
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
int csr_to_dense(const CSRMatrix* csr, float* dense);

// GPU 数据传输
int csr_to_gpu(CSRMatrix* mat);
int csr_from_gpu(CSRMatrix* mat);
void csr_free_gpu(CSRMatrix* mat);

// 元素访问
float csr_get_element(const CSRMatrix* mat, int row, int col);

// 序列化
int csr_serialize(const CSRMatrix* mat, const char* filename);
int csr_deserialize(CSRMatrix* mat, const char* filename);  // 注意：in-place 风格

// 统计与验证
CSRStats csr_compute_stats(const CSRMatrix* mat);
bool csr_validate(const CSRMatrix* mat);
```

    ## #CSRStats 结构

```cpp struct CSRStats {
    float avg_nnz_per_row;  // 平均每行非零元素数
    int max_nnz_per_row;    // 最大每行非零元素数
    int min_nnz_per_row;    // 最小每行非零元素数
    float skewness;         // 倾斜度: max / (min + 1)
};
```

    -- -

    ##ELL 矩阵

    ## #数据结构

```cpp struct ELLMatrix {
    int num_rows;         // 行数
    int num_cols;         // 列数
    int max_nnz_per_row;  // 每行最大非零元素数（决定填充）
    int nnz;              // 实际非零元素总数

    // 列主序存储: values[k * num_rows + row]
    float* values;     // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;  // 列索引数组，-1 表示填充

    // GPU 设备指针
    float* d_values;
    int* d_col_indices;

    // 内存所有权标志
    bool owns_host_memory;
    bool owns_device_memory;
};
```

    ## #核心函数

```cpp
    // 创建与销毁
    ELLMatrix*
    ell_create(int rows, int cols, int max_nnz_per_row);
void ell_destroy(ELLMatrix* mat);

// 数据转换
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
int ell_to_dense(const ELLMatrix* ell, float* dense);

// GPU 数据传输
int ell_to_gpu(ELLMatrix* mat);
int ell_from_gpu(ELLMatrix* mat);
void ell_free_gpu(ELLMatrix* mat);

// 元素访问
float ell_get_element(const ELLMatrix* mat, int row, int col);

// 序列化
int ell_serialize(const ELLMatrix* mat, const char* filename);
int ell_deserialize(ELLMatrix* mat, const char* filename);

// 验证
bool ell_validate(const ELLMatrix* mat);
```

    -- -

    ##SpMV 计算

    ## #Kernel 类型

```cpp enum KernelType {
        SCALAR_CSR,  // 1 线程/行 - 适合极稀疏矩阵
        VECTOR_CSR,  // 1 Warp/行 - 适合均匀分布矩阵
        MERGE_PATH,  // 负载均衡 - 适合倾斜矩阵
        ELL_KERNEL   // ELL 格式专用
    };
```

    ## #配置结构

```cpp struct SpMVConfig {
    KernelType kernel_type;
    int block_size;    // CUDA 块大小 (默认 256)
    bool use_texture;  // 是否使用纹理缓存
};

// 阈值配置（用于自动选择）
struct SpMVThresholds {
    float avg_nnz_threshold;     // 默认 4.0
    float skewness_threshold;    // 默认 10.0
    int texture_cols_threshold;  // 默认 10000
};
```

    ## #执行上下文（可选，用于纹理缓存复用）

```cpp struct SpMVExecutionContext {
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

    ## #结果结构

```cpp struct SpMVResult {
    float* y;              // 输出向量（设备指针）
    float elapsed_ms;      // 执行时间（毫秒）
    float gflops;          // 计算性能 (GFLOPS)
    float bandwidth_gb_s;  // 内存带宽 (GB/s)
    int error_code;        // 0 = 成功，负数 = 错误
};
```

    ## #核心函数

```cpp
        // 自动选择最优配置
        SpMVConfig
        spmv_auto_config(const CSRMatrix* A);

// CSR SpMV（GPU）
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr,
                    int vec_size = -1,  // -1 表示自动检测
                    SpMVExecutionContext* context = nullptr);

// ELL SpMV（GPU）
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config = nullptr, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);

// CPU 参考实现（用于验证）
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

// 阈值配置
SpMVThresholds spmv_get_thresholds();
void spmv_set_thresholds(const SpMVThresholds& thresholds);
```

    -- -

    ##RAII 内存管理

```cpp template <typename T>
    class CudaBuffer {
   public:
    explicit CudaBuffer(size_t count);
    ~CudaBuffer();  // 自动释放 GPU 内存

    T* data();
    const T* data() const;
    size_t size() const;

    // 禁用拷贝
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // 允许移动
    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
};
```

    -- -

    ##PageRank

    ## #配置与结果

```cpp struct PageRankConfig {
    float damping_factor;  // 阻尼因子（默认 0.85）
    float tolerance;       // 收敛阈值（默认 1e-6）
    int max_iterations;    // 最大迭代次数（默认 100）
};

struct PageRankResult {
    float* ranks;          // PageRank 分数 [num_nodes]
    int iterations;        // 实际迭代次数
    float final_residual;  // 最终残差
    bool converged;        // 是否收敛
    int error_code;        // 0 = 成功
};

struct TopKNode {
    int node_id;  // 节点 ID
    float rank;   // PageRank 分数
};
```

    ## #核心函数

```cpp
        // 计算 PageRank
        PageRankResult
        pagerank(const CSRMatrix* adj_matrix, const PageRankConfig* config = nullptr);

// 获取 Top-K 节点
void pagerank_top_k(const PageRankResult* result, int num_nodes, int k, TopKNode* top_k);

// 释放结果内存
void pagerank_free(PageRankResult* result);
```

    -- -

    ##完整示例

```cpp
#include <spmv/spmv.h>

    int
    main() {
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

    -- -

    <div class = "text-center text-small" style = "margin-top: 3rem;">
    <p> 更多示例见<a href = "examples"> 示例代码</ a> 页面</ p></ div>
