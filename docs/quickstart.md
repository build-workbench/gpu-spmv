---
layout: default
title: 快速开始
nav_order: 2
permalink: /quickstart
lang: zh
---

<p align="right">
  <a href="quickstart.en">🇺🇸 English</a>
</p>

# 快速开始
{: .no_toc }

本文档将帮助您在 5 分钟内完成 GPU SpMV 的安装和第一个程序运行。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|:-----|:--------:|:--------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ 编译器 | GCC 7+ / MSVC 2019+ | GCC 11+ / MSVC 2022+ |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |
| GPU 显存 | 4 GB | 8 GB+ |

### 检查 CUDA 安装

```bash
nvcc --version
nvidia-smi
```

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
```

### 2. 构建项目

使用 CMake Presets（推荐）：

```bash
# Release 模式构建
cmake --preset release
cmake --build --preset release
```

或使用传统方式：

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. 运行测试

```bash
# 运行所有测试
ctest --preset default

# 或直接运行测试程序
./build-release/spmv_tests
```

---

## 第一个程序

创建 `first_spmv.cpp`：

```cpp
#include <spmv/spmv.h>
#include <cstdio>

int main() {
    // 创建 3x3 稀疏矩阵: [1 0 2; 0 3 4; 0 0 5]
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, dense, 3, 3);
    csr_to_gpu(csr);
    
    // 准备输入向量 x = [1, 1, 1]
    float h_x[] = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // 执行 SpMV: y = A * x
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    
    if (result.error == SpMVError::SUCCESS) {
        printf("Success! Time: %.3f ms\n", result.time_ms);
        
        // 读取结果
        float h_y[3];
        cudaMemcpy(h_y, d_y.data(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Result: [%.0f, %.0f, %.0f]\n", h_y[0], h_y[1], h_y[2]);
        // 输出: [3, 7, 5]
    }
    
    csr_destroy(csr);
    return 0;
}
```

### 编译运行

```bash
# 编译
nvcc -o first_spmv first_spmv.cpp \
    -I./include \
    -L./build-release -lgpu_spmv \
    -lcudart

# 运行
./first_spmv
```

---

## 下一步

- **[API 参考](api)** - 了解完整接口
- **[示例代码](examples)** - 更多使用示例
- **[性能优化](performance)** - 调优指南

---

## 常见问题

### Q: 构建失败，找不到 CUDA？

确保 CUDA 已正确安装并设置了环境变量：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: 测试失败？

检查 GPU 是否可用：

```bash
nvidia-smi
```

如果无 GPU，使用 CPU-only 模式测试：

```bash
cmake --preset minimal
```

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>遇到问题？<a href="https://github.com/LessUp/gpu-spmv/issues">提交 Issue</a></p>
</div>
