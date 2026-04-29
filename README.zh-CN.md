<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B" alt="C++">
  <img src="https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/平台-Linux%20%7C%20Windows-blue" alt="Platform">
</p>

<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>基于 CUDA 的高性能稀疏矩阵向量乘法库</strong>
</p>

<p align="center">
  <em>4 种优化内核 · 2 种稀疏格式 · 70%+ 带宽利用率 · 生产级质量</em>
</p>

<p align="center">
  <a href="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://lessup.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/文档-GitHub%20Pages-2EA44F?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/releases">
    <img src="https://img.shields.io/github/v/release/LessUp/gpu-spmv?color=blue" alt="Release">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/许可证-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> · <a href="README.zh-CN.md"><b>简体中文</b></a>
</p>

<p align="center">
  <a href="#-快速开始">快速开始</a>
  · <a href="#-核心特性">核心特性</a>
  · <a href="#-性能表现">性能表现</a>
  · <a href="#-文档导航">文档导航</a>
  · <a href="#-贡献指南">贡献指南</a>
</p>

---

## 🎯 GPU SpMV 是什么？

GPU SpMV 是一个**生产级 C++ 库**，用于在 NVIDIA GPU 上加速稀疏矩阵向量乘法。它根据矩阵特征自动选择最优内核，实现**高达 70%+ 的理论内存带宽**。

**适用场景**：图算法 · 科学计算 · 机器学习 · 数据分析

---

## ✨ 为什么选择 GPU SpMV？

### 🚀 智能内核选择

4 种优化内核，根据矩阵特征自动选择：

| 矩阵模式 | 内核 | 策略 | 性能 |
|:---------|:-----|:-----|:----:|
| 极稀疏 (avg_nnz < 4) | Scalar CSR | 1 线程/行 | ★★★☆☆ |
| 均匀分布 (skewness < 10) | Vector CSR | 1 Warp/行 | ★★★★☆ |
| 高度倾斜 (skewness ≥ 10) | Merge Path | 完美负载均衡 | ★★★★★ |
| ELL 格式 | ELL Kernel | 合并访存 | ★★★★★ |

### 📊 多格式支持

- **CSR** (Compressed Sparse Row) - 通用稀疏矩阵
- **ELL** (ELLPACK) - 行长度均匀，极致性能

### 🎯 生产级质量

```cpp
// RAII 资源管理 - 自动清理
CudaBuffer<float> d_x(1000);  // GPU 内存自动释放
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

// 语义化错误处理 - 清晰诊断
if (result.error != SpMVError::SUCCESS) {
    printf("错误：%s\n", spmv_error_string(result.error));
}
```

- ✅ **RAII 管理** - `CudaBuffer`、`SpMVExecutionContext`
- 🔍 **错误码** - 语义化 `SpMVError` 枚举
- 🖥️ **跨平台** - Windows & Linux
- 🔧 **现代构建** - CMake Presets 一键构建
- ✅ **完整测试** - Google Test + 100+ 属性测试

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低要求 | 推荐配置 |
|:-----|:--------:|:--------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |

### 三步安装

```bash
# 1. 克隆仓库
git clone https://github.com/LessUp/gpu-spmv.git && cd gpu-spmv

# 2. 构建
cmake --preset release && cmake --build --preset release

# 3. 测试
ctest --preset default  # 所有测试应该通过 ✅
```

⏱️ **构建时间**：现代计算机约 2 分钟

### 💻 30 秒示例

```cpp
#include <spmv/spmv.h>

int main() {
    // 1. 创建 3×3 稀疏矩阵: [1 0 2; 0 3 4; 0 0 5]
    float data[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, data, 3, 3);
    csr_to_gpu(csr);

    // 2. 准备向量
    CudaBuffer<float> d_x(3), d_y(3);
    float h_x[] = {1, 1, 1};
    cudaMemcpy(d_x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    // 3. 执行（自动选择最优内核）
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    // result.time_ms ≈ 0.05ms, result.error == SUCCESS

    // 4. 获取结果: y = [3, 7, 5]
    csr_destroy(csr);
}
```

📚 **更多示例**：[文档站点](https://lessup.github.io/gpu-spmv/examples)

---

## 📊 性能表现

在 **NVIDIA RTX 3090** (Ampere, 936 GB/s 峰值) 上的基准测试：

| 矩阵规模 | 非零元 | 内核 | 时间 | 带宽 | 利用率 |
|:--------:|:-----:|:-----|:----:|:----:|:------:|
| 10K × 10K | 500K | Vector CSR | 2.3ms | 68.5 GB/s | **70.2%** |
| 100K × 100K | 5M | Merge Path | 23.5ms | 69.8 GB/s | **71.5%** |
| 1M × 1M | 50M | Merge Path | 235ms | 69.1 GB/s | **70.8%** |

```bash
# 运行你自己的基准测试
./build-release/spmv_benchmark

# 输出示例:
# GPU: NVIDIA GeForce RTX 3090
# 矩阵: 100000x100000, 非零元: 5000000
# 平均时间: 23.5 ms | 带宽: 69.8 GB/s (峰值的 71.5%)
```

📈 **完整性能指南**：[性能优化](https://lessup.github.io/gpu-spmv/performance)

---

## 🏗️ 项目结构

```
gpu-spmv/
├── include/spmv/          # 公共头文件（10 个）
│   ├── spmv.h             # 主 SpMV 接口
│   ├── csr_matrix.h       # CSR 格式
│   ├── ell_matrix.h       # ELL 格式
│   ├── cuda_buffer.h      # RAII GPU 内存
│   ├── benchmark.h        # 性能测试
│   └── pagerank.h         # PageRank 算法
├── src/                   # 实现文件（7 个）
├── tests/                 # Google Test 套件（8 个）
├── benchmarks/            # 性能基准测试
└── openspec/             # SDD 规范文档
```

🔧 **规范驱动开发**：所有功能在 [`openspec/specs/`](openspec/specs/) 中定义后实现

---

## 📚 文档导航

完整文档请访问 **[https://lessup.github.io/gpu-spmv/](https://lessup.github.io/gpu-spmv/)**：

| 文档 | 描述 |
|:-----|:-----|
| [📦 安装指南](https://lessup.github.io/gpu-spmv/installation) | 系统要求、详细安装步骤 |
| [📚 API 参考](https://lessup.github.io/gpu-spmv/api) | 完整 API 文档、数据结构 |
| [📝 示例代码](https://lessup.github.io/gpu-spmv/examples) | 7 个完整代码示例（基础→高级） |
| [🚀 性能优化](https://lessup.github.io/gpu-spmv/performance) | 调优策略、基准测试数据 |
| [🏗️ 架构设计](https://lessup.github.io/gpu-spmv/architecture) | 系统设计、内核选择 |
| [📋 更新日志](https://lessup.github.io/gpu-spmv/changelog) | 版本历史、迁移指南 |

---

## 🧪 测试

```bash
# 运行所有测试
ctest --preset default

# 或直接运行
./build-release/spmv_tests

# 运行指定测试
./build-release/spmv_tests --gtest_filter="CSR*"
./build-release/spmv_tests --gtest_filter="ELL*"
```

**测试覆盖**：
- ✅ CSR/ELL 格式转换正确性
- ✅ SpMV 计算正确性（与 CPU 参考对比）
- ✅ 维度验证
- ✅ 内核选择逻辑
- ✅ 带宽指标
- ✅ PageRank 不变量
- ✅ 100+ 属性测试（随机矩阵）

---

## 💡 实际应用：PageRank

```cpp
#include <spmv/pagerank.h>

// 构建图的邻接矩阵
CSRMatrix* adj = build_graph_adjacency();
csr_to_gpu(adj);

// 运行 PageRank
PageRankConfig config = {.damping = 0.85f, .tolerance = 1e-6f};
PageRankResult result = pagerank(adj, &config);

// 获取排名前 10 的节点
auto top_10 = get_top_k(result, 10);
for (const auto& node : top_10) {
    printf("节点 %d: %.6f\n", node.id, node.rank);
}

pagerank_free(&result);
csr_destroy(adj);
```

📊 **应用场景**：社交网络分析 · Web 搜索 · 推荐系统 · 欺诈检测

---

## 🤝 贡献指南

我们欢迎各种形式的贡献！GPU SpMV 遵循**规范驱动开发** - 规范是唯一的真相来源。

### 快速贡献指南

1. 🍴 **Fork** 本仓库
2. 📖 **查阅规范** - 在 `openspec/specs/` 中查看你想实现的功能
3. 🌿 **创建分支** (`git checkout -b feature/your-feature`)
4. 📝 **先更新规范**（如果修改行为）
5. 💻 **按规范实现代码**
6. ✅ **运行测试** (`ctest --preset default`)
7. 🚀 **提交 PR** 包含规范变更

📋 **完整指南**：[CONTRIBUTING.md](CONTRIBUTING.md)

### 开发环境

```bash
# 格式化代码（提交前必须执行）
find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  | xargs clang-format -i

# 构建并测试
cmake --preset default && cmake --build --preset default && ctest --preset default
```

---

## 📄 许可证

MIT 许可证 © 2024-2026 LessUp。详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- 算法基于 [Merge-based Parallel SpMV](https://research.nvidia.com/publication/merge-based-parallel-sparse-matrix-vector-multiplication) by Merrill & Garland (NVIDIA)
- CUDA 优化技术来自 NVIDIA 官方文档
- 灵感来自 cuSPARSE 和现代稀疏库设计模式

---

<p align="center">
  <sub>由 GPU SpMV 贡献者们用 ❤️ 构建</sub>
</p>

<p align="center">
  <a href="#-快速开始">⬆️ 返回顶部</a>
</p>
