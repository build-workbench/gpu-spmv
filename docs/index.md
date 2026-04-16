---
layout: default
title: 首页
---

# GPU SpMV

**基于 CUDA 的高性能稀疏矩阵向量乘法库**

支持 CSR 和 ELL 格式，包含多种负载均衡优化策略，自动选择最优 Kernel。

[![GitHub](https://img.shields.io/badge/GitHub-LessUp/gpu--spmv-blue?logo=github)](https://github.com/LessUp/gpu-spmv)
[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)

---

## 核心特性

### 多种稀疏矩阵格式

| 格式 | 描述 | 适用场景 |
|------|------|----------|
| **CSR** | Compressed Sparse Row | 通用格式，适合大多数稀疏矩阵 |
| **ELL** | ELLPACK | 行长度均匀的矩阵，GPU 访存最优 |

### 四种优化 CUDA Kernel

```
┌─────────────────────────────────────────────────────────────┐
│                    Kernel 自动选择策略                        │
├─────────────────────────────────────────────────────────────┤
│  avg_nnz_per_row < 4     ──→  Scalar CSR                    │
│  (每行非零元素少)              一个线程处理一行               │
│                                                             │
│  skewness < 10           ──→  Vector CSR                    │
│  (行长度较均匀)                一个 Warp 协作处理一行          │
│                                                             │
│  skewness >= 10          ──→  Merge Path                    │
│  (行长度差异大)                工作量完全均匀分配              │
│                                                             │
│  行长度接近 max_nnz       ──→  ELL Kernel                    │
│  (ELL 格式)                   Column-major 合并访问          │
└─────────────────────────────────────────────────────────────┘
```

### 工程质量

- **RAII 资源管理** — `CudaBuffer`、`CudaTimer`、`ScopedTexture` 自动管理 GPU 资源
- **语义化错误码** — `SpMVError` 枚举 + `CUDA_CHECK_*` 宏，清晰的错误追踪
- **跨平台支持** — Windows / Linux 测试路径自动适配
- **现代构建系统** — CMake Presets，一键 Debug/Release 构建
- **持续集成** — GitHub Actions 自动格式检查和构建验证

---

## 快速开始

### 环境要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU (Compute Capability 7.0+)

### 安装构建

```bash
# 克隆仓库
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# Release 构建（推荐）
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset default
```

### 30 秒示例

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

int main() {
    // 1. 创建稀疏矩阵
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    // 2. 准备输入向量
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);

    // 3. 执行 SpMV（自动选择最优 Kernel）
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

    // 4. 获取结果
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);

    // 输出: y = [3, 7, 5]
    printf("Result: [%.0f, %.0f, %.0f]\n", y[0], y[1], y[2]);
    
    csr_destroy(csr);
    return 0;
}
```

---

## 应用示例

### PageRank 图算法

```cpp
#include "spmv/pagerank.h"

// 创建列归一化的邻接矩阵
CSRMatrix* adj = create_adjacency_matrix();
csr_to_gpu(adj);

// 配置 PageRank 参数
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;
config.max_iterations = 100;

// 运行 PageRank
PageRankResult result = pagerank(adj, &config);

// 获取 Top-10 节点
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

### 性能基准测试

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig config;
config.num_warmup_runs = 5;
config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x.data(), nullptr, &config);

printf("Average time: %.3f ms\n", result.avg_time_ms);
printf("GFLOPS: %.2f\n", result.gflops);
printf("Bandwidth: %.1f GB/s (%.1f%% efficiency)\n", 
       result.bandwidth_gb_s,
       result.bandwidth_gb_s / get_gpu_peak_bandwidth() * 100);
```

---

## 文档导航

| 文档 | 描述 |
|------|------|
| [**API 参考**](api) | 完整的 API 文档：数据结构、函数接口、错误码 |
| [**性能优化**](performance) | Kernel 选择策略、带宽优化技巧、基准测试指南 |
| [**示例代码**](examples) | 完整示例：基本用法、格式转换、PageRank 应用 |
| [**更新日志**](changelog) | 版本历史、变更记录、迁移指南 |

---

## 性能概览

在典型稀疏矩阵上的性能表现：

| 矩阵规模 | 非零元素 | Kernel | 带宽利用率 |
|----------|----------|--------|------------|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

> 实际性能取决于矩阵结构和 GPU 型号。使用 `spmv_auto_config()` 自动选择最优策略。

---

## 项目结构

```
gpu-spmv/
├── include/spmv/          # 公共头文件
│   ├── common.h           # 错误码、CUDA 检查宏
│   ├── cuda_buffer.h      # RAII GPU 内存管理
│   ├── csr_matrix.h       # CSR 稀疏矩阵
│   ├── ell_matrix.h       # ELL 稀疏矩阵
│   ├── spmv.h             # SpMV 接口、Kernel 选择
│   ├── bandwidth.h        # 带宽度量
│   ├── benchmark.h        # 基准测试框架
│   └── pagerank.h         # PageRank 算法
├── src/                   # 源实现
├── tests/                 # 属性测试 + 单元测试
├── benchmarks/            # 性能基准测试
└── docs/                  # 在线文档
```

---

## 许可证

[MIT License](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)

---

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request
