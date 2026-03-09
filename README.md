# GPU SpMV (稀疏矩阵向量乘法)

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/gpu-spmv/)

简体中文 | [English](README.en.md)

基于 CUDA 的高性能稀疏矩阵向量乘法库，支持 CSR 和 ELL 格式，包含多种负载均衡优化策略。

## 特性

- **多种稀疏矩阵格式**
  - CSR (Compressed Sparse Row)
  - ELL (ELLPACK)

- **优化的 CUDA Kernels**
  - Scalar CSR: 一个线程处理一行
  - Vector CSR: 一个 Warp (32线程) 处理一行
  - Merge Path: 工作量均匀分配，适合高度不均匀的矩阵
  - ELL Kernel: Column-major 访问，适合均匀行长度

- **自动 Kernel 选择**
  - 基于矩阵特征自动选择最优 Kernel

- **性能度量**
  - 带宽利用率分析（峰值带宽缓存，避免重复查询）
  - GFLOPS 计算
  - 完整的基准测试框架（CPU 计时使用 `std::chrono`）
  - 可选纹理缓存读取输入向量

- **工程质量**
  - RAII 资源管理（`CudaBuffer`、`CudaTimer`、`ScopedTexture`）
  - 语义化错误码（`CUDA_CHECK_MALLOC` / `CUDA_CHECK_MEMCPY`）
  - 跨平台测试路径（Windows / Linux）
  - CMake Presets 支持
  - `.clang-format` + `.editorconfig` 代码风格
  - GitHub Actions CI

- **应用示例**
  - PageRank 图算法实现

## 构建

### 要求

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

# 最小构建（仅 sm_80）
cmake --preset minimal
cmake --build --preset minimal
```

### 手动构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### 运行测试

```bash
ctest --preset default
# 或直接运行
./build/spmv_tests
```

### 运行基准测试

```bash
./build/spmv_benchmark
```

## 使用示例

### 基础 SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

// 创建稠密矩阵
std::vector<float> dense = {
    1, 0, 2,
    0, 3, 4,
    0, 0, 5
};

// 转换为 CSR 格式
CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, dense.data(), 3, 3);
csr_to_gpu(csr);

// 输入向量
std::vector<float> x = {1, 1, 1};
CudaBuffer<float> d_x(3);
CudaBuffer<float> d_y(3);
d_x.copyFromHost(x.data(), 3);

// 执行 SpMV
SpMVConfig config = spmv_auto_config(csr);
// config.use_texture = true;  // 可选：启用纹理缓存读取 x
SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

// 获取结果
std::vector<float> y(3);
d_y.copyToHost(y.data(), 3);

csr_destroy(csr);
```

### PageRank

```cpp
#include "spmv/pagerank.h"

// 创建邻接矩阵 (列归一化)
CSRMatrix* adj = /* ... */;
csr_to_gpu(adj);

// 运行 PageRank
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

PageRankResult result = pagerank(adj, &config);

// 获取 Top-10 节点
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

### 基准测试

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig bench_config;
bench_config.num_warmup_runs = 5;
bench_config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x.data(), &spmv_config, &bench_config);

std::cout << "Avg time: " << result.avg_time_ms << " ms\n";
std::cout << "GFLOPS: " << result.gflops << "\n";
std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";

// 导出 JSON
std::string json = benchmark_to_json(result);
```

## 性能优化

### Kernel 选择策略

- **短行 (avg_nnz < 4)**: Scalar CSR
- **均匀分布 (skewness < 10)**: Vector CSR
- **高度不均匀 (skewness >= 10)**: Merge Path

### 带宽优化

- Column-major 存储 (ELL 格式)
- 合并内存访问
- 带宽利用率监控

## 测试

项目包含完整的属性测试套件，验证：

- CSR/ELL 格式转换正确性
- SpMV 计算正确性 (与 CPU 参考对比)
- 维度验证
- Kernel 选择器有效性
- 带宽度量有效性
- PageRank 不变量

每个属性测试运行 100 次迭代，使用随机生成的矩阵。

## 项目结构

```
.
├── include/spmv/       # 头文件
│   ├── common.h        # 错误码、CUDA_CHECK_* 宏
│   ├── cuda_buffer.h   # RAII GPU 内存 (memset/fill/bytes)
│   ├── csr_matrix.h    # CSR 稀疏矩阵
│   ├── ell_matrix.h    # ELL 稀疏矩阵 (含 nnz 字段)
│   ├── spmv.h          # SpMV 接口与自动 Kernel 选择
│   ├── bandwidth.h     # 带宽度量
│   ├── benchmark.h     # 基准测试框架
│   ├── pagerank.h      # PageRank 算法
│   └── test_utils.h    # 测试工具 (跨平台路径等)
├── src/                # 源文件
│   ├── csr_matrix.cpp
│   ├── ell_matrix.cpp
│   ├── spmv_cpu.cpp
│   ├── spmv_kernels.cu  # RAII helpers: CudaTimer, ScopedTexture
│   ├── bandwidth.cpp    # 峰值带宽缓存 (std::call_once)
│   ├── benchmark.cu     # CPU 计时使用 std::chrono
│   └── pagerank.cu
├── tests/              # 属性测试 + 单元测试
├── benchmarks/         # 基准测试程序
├── .clang-format       # 代码风格
├── .editorconfig       # 编辑器配置
├── CMakePresets.json   # 构建预设
└── .github/workflows/  # CI
```

## 在线文档

项目文档已通过 GitHub Pages 发布：

> **https://lessup.github.io/gpu-spmv/**

包含：
- [API 参考](https://lessup.github.io/gpu-spmv/api) — 头文件接口、数据结构与函数说明
- [性能优化](https://lessup.github.io/gpu-spmv/performance) — Kernel 选择策略、带宽优化与基准测试

## 许可证

MIT License
