---
layout: default
title: GPU SpMV 文档
lang: zh
---

<p align="right">
  <a href="index.en.html">🇺🇸 English</a>
</p>

# GPU SpMV 文档中心

**基于 CUDA 的高性能稀疏矩阵向量乘法库**

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](https://github.com/LessUp/gpu-spmv/releases)

---

## 快速导航

| 文档 | 描述 | 适用读者 |
|:----:|:-----|:---------|
| [**📦 安装指南**](installation) | 系统要求、构建步骤、依赖安装 | 新用户 |
| [**📚 API 参考**](api) | 完整接口文档、数据结构、错误处理 | 开发者 |
| [**📝 示例代码**](examples) | 基础用法、高级特性、完整应用 | 学习者 |
| [**🚀 性能优化**](performance) | 调优策略、基准测试、最佳实践 | 性能工程师 |
| [**🏗️ 架构设计**](architecture) | 系统架构、核心算法、设计决策 | 架构师 |
| [**📋 更新日志**](changelog) | 版本历史、迁移指南、路线图 | 维护者 |

---

## 项目简介

GPU SpMV 是一个专为 NVIDIA GPU 优化的**稀疏矩阵向量乘法 (Sparse Matrix-Vector Multiplication)** 计算库。本库支持多种稀疏矩阵存储格式，通过智能化的内核选择策略，在各种矩阵分布特性下均能达到优异的内存带宽利用率。

### 核心优势

<div class="feature-grid">

**🎯 多格式支持**
- CSR (Compressed Sparse Row) - 通用格式
- ELL (ELLPACK) - GPU 优化格式

**⚡ 智能内核选择**
- 基于矩阵统计特征自动选择最优 Kernel
- 支持 Scalar/Vector/Merge Path/ELL 四种策略

**📊 高性能实现**
- Warp 级并行归约
- 纹理缓存优化
- 合并访存模式

**🔧 工程级质量**
- RAII 资源管理
- 语义化错误码
- 完整测试覆盖

</div>

---

## 30 秒快速开始

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

    // 2. 准备输入/输出向量
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);

    // 3. 自动选择最优配置并执行
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

    // 4. 获取结果
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    // 结果: y = [3, 7, 5]

    csr_destroy(csr);
    return 0;
}
```

---

## 性能指标

在 NVIDIA RTX 3080 上的典型性能表现：

| 矩阵规模 | 非零元素 | 推荐 Kernel | 带宽利用率 |
|:--------:|:--------:|:------------|:----------:|
| 10K × 10K | 500K | Vector CSR | **~70%** |
| 100K × 100K | 5M | Merge Path | **~65%** |
| 1M × 1M | 50M | Merge Path | **~60%** |

> 💡 实际性能取决于矩阵结构和 GPU 型号。使用 `spmv_auto_config()` 可自动选择最优策略。

---

## 文档约定

### 符号说明

| 符号 | 含义 |
|:----:|:-----|
| `代码` | 代码片段、函数名、变量名 |
| **粗体** | 重要概念、强调内容 |
| [链接]() | 可点击的文档链接 |
| ⚠️ 警告 | 需要特别注意的事项 |
| 💡 提示 | 有用的提示和建议 |
| 📝 说明 | 补充说明信息 |

### 版本信息

本文档对应 GPU SpMV **v1.0.0**。如需查看其他版本的文档，请访问 [GitHub Releases](https://github.com/LessUp/gpu-spmv/releases)。

---

## 获取帮助

- **问题反馈**: [GitHub Issues](https://github.com/LessUp/gpu-spmv/issues)
- **功能建议**: [GitHub Discussions](https://github.com/LessUp/gpu-spmv/discussions)
- **源码仓库**: [GitHub Repository](https://github.com/LessUp/gpu-spmv)

---

## 许可证

本项目采用 [MIT License](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE) 开源许可证。

---

<div align="center">

**[开始阅读 📚](installation)** · **[查看示例 📝](examples)** · **[GitHub 仓库 🐙](https://github.com/LessUp/gpu-spmv)**

</div>
