---
layout: default
title: 首页
---

# GPU SpMV

基于 CUDA 的高性能稀疏矩阵向量乘法库，支持 CSR 和 ELL 格式，包含多种负载均衡优化策略。

## 特性

- **多种稀疏矩阵格式** — CSR (Compressed Sparse Row)、ELL (ELLPACK)
- **优化的 CUDA Kernels** — Scalar CSR、Vector CSR、Merge Path、ELL Kernel
- **自动 Kernel 选择** — 基于矩阵特征自动选择最优 Kernel
- **性能度量** — 带宽利用率分析、GFLOPS 计算、完整的基准测试框架
- **工程质量** — RAII 资源管理、语义化错误码、CMake Presets、CI

## 快速开始

### 构建要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU (Compute Capability 7.0+)

### 使用 CMake Presets（推荐）

```bash
# Debug 构建
cmake --preset default
cmake --build --preset default

# Release 构建
cmake --preset release
cmake --build --preset release
```

### 基础 SpMV 示例

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

// 创建稠密矩阵并转换为 CSR 格式
std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, dense.data(), 3, 3);
csr_to_gpu(csr);

// 输入向量
std::vector<float> x = {1, 1, 1};
CudaBuffer<float> d_x(3), d_y(3);
d_x.copyFromHost(x.data(), 3);

// 执行 SpMV（自动选择最优 Kernel）
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

// 获取结果
std::vector<float> y(3);
d_y.copyToHost(y.data(), 3);
csr_destroy(csr);
```

## 文档导航

- [**API 参考**](api) — 头文件接口、数据结构与函数说明
- [**性能优化**](performance) — Kernel 选择策略、带宽优化与基准测试

## 项目结构

```
├── include/spmv/       # 头文件
│   ├── common.h        # 错误码、CUDA_CHECK_* 宏
│   ├── cuda_buffer.h   # RAII GPU 内存
│   ├── csr_matrix.h    # CSR 稀疏矩阵
│   ├── ell_matrix.h    # ELL 稀疏矩阵
│   ├── spmv.h          # SpMV 接口与自动 Kernel 选择
│   ├── bandwidth.h     # 带宽度量
│   ├── benchmark.h     # 基准测试框架
│   └── pagerank.h      # PageRank 算法
├── src/                # 源文件
├── tests/              # 属性测试 + 单元测试
└── benchmarks/         # 基准测试程序
```

## 许可证

[MIT License](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)
