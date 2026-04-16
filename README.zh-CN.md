<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>基于 CUDA 的高性能稀疏矩阵向量乘法库</strong>
</p>

<p align="center">
  <a href="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://lessup.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/文档-GitHub%20Pages-blue?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/releases">
    <img src="https://img.shields.io/badge/版本-1.0.0-blue.svg" alt="Version">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/许可证-MIT-green.svg" alt="License">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

<p align="center">
  支持格式: <b>CSR</b>, <b>ELL</b> | 内核: <b>Scalar</b>, <b>Vector</b>, <b>Merge Path</b> | 自动选择 | 生产就绪
</p>

---

## ✨ 核心特性

### 多种稀疏矩阵格式

| 格式 | 描述 | 适用场景 | GPU 效率 |
|:-----|:-----|:---------|:--------:|
| **CSR** | Compressed Sparse Row | 通用稀疏矩阵 | ★★★☆☆ |
| **ELL** | ELLPACK | 行长度均匀的矩阵 | ★★★★★ |

### 四种优化的 CUDA Kernel

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kernel 自动选择策略                          │
├─────────────────────────────────────────────────────────────────┤
│  avg_nnz_per_row < 4     ──→  Scalar CSR                        │
│  (每行非零元素很少)            一个线程处理一行                    │
│                                                                 │
│  skewness < 10           ──→  Vector CSR                        │
│  (分布均匀)                    一个 Warp (32 线程) 协作处理一行    │
│                                                                 │
│  skewness >= 10          ──→  Merge Path                        │
│  (分布极度不均匀)              完美的负载均衡                      │
│                                                                 │
│  ELL 格式                ──→  ELL Kernel                        │
│  (行长度接近)                  Column-major 合并访存               │
└─────────────────────────────────────────────────────────────────┘
```

### 生产级工程质量

- 🎯 **RAII 资源管理** — `CudaBuffer`、`CudaTimer`、`SpMVExecutionContext` 自动管理 GPU 资源
- 🔍 **语义化错误码** — `SpMVError` 枚举 + `CUDA_CHECK_*` 宏，清晰的错误追踪
- 🖥️ **跨平台支持** — 自动适配 Windows/Linux 测试路径
- 🔧 **现代构建系统** — CMake Presets 一键构建 Debug/Release
- ✅ **CI/CD 就绪** — GitHub Actions 格式检查和构建验证

---

## 🚀 快速开始

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|:-----|:--------:|:--------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ 标准 | C++17 | C++17 |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |

### 安装构建

```bash
# 克隆仓库
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# 构建 (推荐 Release 模式)
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
    // 1. 从稠密矩阵创建 CSR 格式
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

    csr_destroy(csr);
    return 0;
}
```

---

## 📊 性能表现

NVIDIA RTX 3080 上的基准测试结果：

| 矩阵规模 | 非零元素数 | Kernel | 带宽利用率 |
|:--------:|:----------:|:-------|:----------:|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

```bash
# 运行基准测试
./build-release/spmv_benchmark

# 示例输出:
# GPU: NVIDIA GeForce RTX 3080
# 矩阵: 100000x100000, 非零元: 5000000
# 平均时间: 0.312 ms
# 带宽: 495.2 GB/s (峰值 65.1%)
```

---

## 📚 文档导航

| 文档 | 描述 |
|:-----|:-----|
| [📖 **文档站点**](https://lessup.github.io/gpu-spmv/) | 完整的技术文档和示例 |
| [📦 **安装指南**](https://lessup.github.io/gpu-spmv/installation) | 详细的安装步骤说明 |
| [📚 **API 参考**](https://lessup.github.io/gpu-spmv/api) | 完整的 API 接口文档 |
| [🚀 **性能优化**](https://lessup.github.io/gpu-spmv/performance) | 性能调优策略指南 |
| [📝 **示例代码**](https://lessup.github.io/gpu-spmv/examples) | 丰富的代码示例集合 |
| [📋 **更新日志**](https://lessup.github.io/gpu-spmv/changelog) | 版本历史和迁移指南 |

---

## 🏗️ 项目结构

```
gpu-spmv/
├── include/spmv/          # 公共头文件
│   ├── common.h           # 错误码、CUDA 宏
│   ├── cuda_buffer.h      # RAII GPU 内存管理
│   ├── csr_matrix.h       # CSR 稀疏矩阵格式
│   ├── ell_matrix.h       # ELL 稀疏矩阵格式
│   ├── spmv.h             # SpMV 接口、Kernel 选择
│   ├── bandwidth.h        # 带宽度量
│   ├── benchmark.h        # 基准测试框架
│   └── pagerank.h         # PageRank 算法
├── src/                   # 源文件实现
├── tests/                 # 属性测试 + 单元测试
├── benchmarks/            # 性能基准测试
└── docs/                  # 在线文档
```

---

## 🧪 测试

```bash
# 运行所有测试
./build-release/spmv_tests

# 运行指定测试
./build-release/spmv_tests --gtest_filter="CSR*"

# 属性测试（随机矩阵）
./build-release/spmv_tests --gtest_repeat=10
```

测试覆盖：
- ✅ CSR/ELL 格式转换正确性
- ✅ SpMV 计算正确性（与 CPU 参考对比）
- ✅ 维度验证
- ✅ Kernel 选择器有效性
- ✅ 带宽度量有效性
- ✅ PageRank 不变量检查

---

## 💡 应用示例: PageRank

```cpp
#include "spmv/pagerank.h"
#include "spmv/csr_matrix.h"

// 创建列归一化的邻接矩阵
CSRMatrix* adj = create_normalized_adjacency();
csr_to_gpu(adj);

// 配置 PageRank 参数
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

// 执行 PageRank
PageRankResult result = pagerank(adj, &config);

// 获取 Top-10 节点
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

---

## 🤝 贡献

我们欢迎各种形式的贡献！详情请参阅 [Contributing Guide](CONTRIBUTING.md)。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 开发环境

```bash
# 格式化代码
find src tests include -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i

# 构建并测试
cmake --preset default
cmake --build --preset default
ctest --preset default
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可。

---

## 🙏 致谢

- 算法灵感源自 [Merge-based Parallel Sparse Matrix-Vector Multiplication](https://research.nvidia.com/sites/default/files/pubs/2014-09_Merge-based-Parallel-Sparse/merge-based-spmv.pdf) by Merrill & Garland
- CUDA 优化技术参考 NVIDIA 官方文档

---

<div align="center">

**[📖 文档站点](https://lessup.github.io/gpu-spmv/)** · **[🚀 快速开始](https://lessup.github.io/gpu-spmv/installation)** · **[💻 GitHub](https://github.com/LessUp/gpu-spmv)**

</div>
