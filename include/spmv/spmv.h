#ifndef SPMV_SPMV_H
#define SPMV_SPMV_H

#include "common.h"
#include "csr_matrix.h"
#include "ell_matrix.h"

namespace spmv {

// SpMV 配置
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

    SpMVConfig() : kernel_type(SCALAR_CSR), block_size(256), use_texture(false) {}
    SpMVConfig(KernelType kernel_type_, int block_size_, bool use_texture_)
        : kernel_type(kernel_type_), block_size(block_size_), use_texture(use_texture_) {}
};

// 可复用的 SpMV 执行上下文
struct SpMVExecutionContext {
    cudaTextureObject_t tex_x;
    const float* cached_x;
    size_t cached_x_length;
    bool texture_enabled;

    SpMVExecutionContext()
        : tex_x(0), cached_x(nullptr), cached_x_length(0), texture_enabled(false) {}

    ~SpMVExecutionContext() { reset(); }

    SpMVExecutionContext(const SpMVExecutionContext&) = delete;
    SpMVExecutionContext& operator=(const SpMVExecutionContext&) = delete;

    SpMVExecutionContext(SpMVExecutionContext&& other) noexcept
        : tex_x(other.tex_x),
          cached_x(other.cached_x),
          cached_x_length(other.cached_x_length),
          texture_enabled(other.texture_enabled) {
        other.tex_x = 0;
        other.cached_x = nullptr;
        other.cached_x_length = 0;
        other.texture_enabled = false;
    }

    SpMVExecutionContext& operator=(SpMVExecutionContext&& other) noexcept {
        if (this != &other) {
            reset();
            tex_x = other.tex_x;
            cached_x = other.cached_x;
            cached_x_length = other.cached_x_length;
            texture_enabled = other.texture_enabled;
            other.tex_x = 0;
            other.cached_x = nullptr;
            other.cached_x_length = 0;
            other.texture_enabled = false;
        }
        return *this;
    }

    void reset() {
        if (tex_x != 0) {
            cudaDestroyTextureObject(tex_x);
            tex_x = 0;
        }
        cached_x = nullptr;
        cached_x_length = 0;
        texture_enabled = false;
    }
};

// SpMV 结果
struct SpMVResult {
    float* y;               // 输出向量 (GPU 或 CPU)
    float elapsed_ms;       // 执行时间
    float gflops;           // 计算吞吐量
    float bandwidth_gb_s;   // 带宽利用率
    int error_code;         // 0 = 成功

    SpMVResult() : y(nullptr), elapsed_ms(0.0f), gflops(0.0f),
                   bandwidth_gb_s(0.0f), error_code(0) {}
};

// CPU 参考实现
void spmv_cpu_csr(const CSRMatrix* A, const float* x, float* y);
void spmv_cpu_ell(const ELLMatrix* A, const float* x, float* y);

// GPU 实现
SpMVResult spmv_csr(const CSRMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);
SpMVResult spmv_ell(const ELLMatrix* A, const float* d_x, float* d_y,
                    const SpMVConfig* config, int vec_size = -1,
                    SpMVExecutionContext* context = nullptr);

// 自动选择最优 Kernel
SpMVConfig spmv_auto_config(const CSRMatrix* A);

// 验证维度
inline bool spmv_validate_dimensions(int num_cols, int vec_size) {
    return num_cols == vec_size;
}

} // namespace spmv

#endif // SPMV_SPMV_H
