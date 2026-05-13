# 基础 SpMV 示例

完整的 GPU SpMV 计算示例。

## 完整示例

```cpp
#include <spmv/spmv.h>
#include <cstdio>
#include <random>

int main() {
    // ===== 1. 矩阵设置 =====
    const int N = 10000;      // 矩阵维度
    const int NNZ = 100000;   // 非零元素数

    // 创建 CSR 矩阵
    CSRMatrix* csr = csr_create(N, N, NNZ);

    // 用随机稀疏数据填充
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> col_dist(0, N - 1);
    std::uniform_real_distribution<float> val_dist(0.0f, 1.0f);

    // 先构建密集表示（简化）
    std::vector<float> dense(N * N, 0.0f);
    for (int i = 0; i < NNZ; i++) {
        int row = i * N / NNZ;
        int col = col_dist(rng);
        dense[row * N + col] = val_dist(rng);
    }

    // 转换为 CSR
    csr_from_dense(csr, dense.data(), N, N);

    // ===== 2. GPU 传输 =====
    csr_to_gpu(csr);

    // ===== 3. 向量设置 =====
    // 输入向量 x = [1, 1, 1, ...]
    std::vector<float> h_x(N, 1.0f);
    CudaBuffer<float> d_x(N), d_y(N);

    cudaMemcpy(d_x.data(), h_x.data(), N * sizeof(float),
               cudaMemcpyHostToDevice);

    // ===== 4. 自动配置 =====
    SpMVConfig config = spmv_auto_config(csr);

    // 打印选择的 Kernel
    const char* kernel_names[] = {
        "Scalar CSR", "Vector CSR", "Merge Path", "ELL Kernel"
    };
    printf("选择的 Kernel: %s\n", kernel_names[config.kernel_type]);

    // ===== 5. 执行 SpMV =====
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config);

    // ===== 6. 检查结果 =====
    if (result.error_code != 0) {
        fprintf(stderr, "错误: %d\n", result.error_code);
        csr_destroy(csr);
        return 1;
    }

    printf("时间: %.3f ms\n", result.elapsed_ms);
    printf("带宽: %.1f GB/s\n", result.bandwidth_gb_s);
    printf("GFLOPS: %.1f\n", result.gflops);

    // ===== 7. 验证（可选）=====
    std::vector<float> h_y_gpu(N), h_y_cpu(N);
    cudaMemcpy(h_y_gpu.data(), d_y.data(), N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // CPU 参考
    spmv_cpu_csr(csr, h_x.data(), h_y_cpu.data());

    // 比较
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = std::abs(h_y_gpu[i] - h_y_cpu[i]);
        max_error = std::max(max_error, error);
    }
    printf("最大误差: %.6e\n", max_error);

    // ===== 8. 清理 =====
    csr_destroy(csr);
    return 0;
}
```

## 编译

```bash
nvcc -o spmv_example spmv_example.cpp \
    -I./include \
    -L./build-release -lgpu_spmv \
    -lcudart
```

## 预期输出

```
选择的 Kernel: Vector CSR
时间: 2.345 ms
带宽: 68.5 GB/s
GFLOPS: 85.6
最大误差: 1.234567e-05
```

## 关键点

1. **使用 `spmv_auto_config()`** 自动选择 Kernel
2. **使用 `CudaBuffer`** 进行 RAII 内存管理
3. **用 CPU 参考验证** 正确性
4. **检查 `error_code`** 进行错误处理
