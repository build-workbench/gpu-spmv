---
layout: default
title: 首页
nav_order: 1
has_children: false
permalink: /
lang: zh
---

# GPU SpMV
{: .fs-9 .fw-700 .text-center }

基于 CUDA 的高性能稀疏矩阵向量乘法库
{: .fs-6 .fw-300 .text-center .text-grey-dk-500 }

<div class="text-center" style="margin: 2rem 0;">
  <a href="installation" class="btn btn-primary fs-5 mb-4 mb-md-0 mr-2">快速开始 →</a>
  <a href="api" class="btn btn-green fs-5 mb-4 mb-md-0 mr-2">API 文档</a>
  <a href="https://github.com/LessUp/gpu-spmv" class="btn fs-5 mb-4 mb-md-0">GitHub</a>
</div>

---

## ✨ 核心特性

<div class="grid grid-cols-3 gap-4" markdown="1">

### 🚀 极致性能

多 Kernel 智能调度，针对 NVIDIA GPU 优化，带宽利用率高达 70%+

### 📊 多格式支持

CSR 通用格式 + ELL 高性能格式，适配不同稀疏性矩阵

### 🎯 生产级质量

RAII 资源管理、语义化错误码、完整测试覆盖，企业级可靠性

</div>

---

## 🔥 为什么选择 GPU SpMV？

### 智能 Kernel 选择

根据矩阵特征自动选择最优 Kernel：

```cpp
// 自动分析矩阵特征，选择最佳 kernel
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

| 矩阵特征 | 推荐 Kernel | 性能 |
|:---------|:-----------|:-----|
| 极稀疏 (avg_nnz < 4) | Scalar CSR | ★★★☆☆ |
| 中等稀疏 (skewness < 10) | Vector CSR | ★★★★☆ |
| 高度倾斜 (skewness ≥ 10) | Merge Path | ★★★★★ |
| ELL 格式 | ELL Kernel | ★★★★★ |

### 简洁的 API 设计

```cpp
#include <spmv/spmv.h>

// 1. 创建 CSR 矩阵
CSRMatrix* csr = csr_create(num_rows, num_cols, nnz);
csr_from_dense(csr, host_data, num_rows, num_cols);

// 2. 传输到 GPU
csr_to_gpu(csr);

// 3. 执行 SpMV
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

// 4. 清理资源
csr_destroy(csr);
```

### 企业级特性

<div class="grid grid-cols-2 gap-3" markdown="1">

#### 🎯 RAII 资源管理

自动生命周期管理，避免内存泄漏：
```cpp
CudaBuffer<float> buffer(1000);
// 自动释放 GPU 内存
```

#### 🔍 语义化错误码

清晰的错误追踪：
```cpp
if (result.error != SpMVError::SUCCESS) {
    printf("Error: %s\n", spmv_error_string(result.error));
}
```

#### 🖥️ 跨平台支持

Windows / Linux 全平台支持

#### 🔧 CMake Presets

一键构建，零配置：
```bash
cmake --preset release
cmake --build --preset release
```

#### ✅ 完整测试覆盖

Google Test + 100+ 属性测试用例

#### 📈 性能基准

内置 benchmark 工具，量化性能指标

</div>

---

## 📊 性能表现

在 NVIDIA RTX 3090 (Ampere) 上的测试结果：

| 矩阵规模 | 非零元素 | Kernel | 带宽利用率 |
|:--------:|:--------:|:-------|:----------:|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

---

## 🚀 快速开始

### 系统要求

- CUDA Toolkit 11.0+ / 12.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU (CC 7.0+)

### 三步安装

```bash
# 1. 克隆仓库
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# 2. 构建
cmake --preset release
cmake --build --preset release

# 3. 测试
ctest --preset default
```

---

## 📚 文档导航

<div class="grid grid-cols-3 gap-4" markdown="1">

### 📦 [安装指南](installation)

系统要求、依赖安装、构建步骤

### 🏗️ [架构设计](architecture)

系统架构、核心算法、设计决策

### 📚 [API 参考](api)

完整接口文档、数据结构、错误处理

### 📝 [示例代码](examples)

基础用法、高级特性、完整应用

### 🚀 [性能优化](performance)

调优策略、基准测试、最佳实践

### 📋 [更新日志](changelog)

版本历史、迁移指南

</div>

---

## 🌟 典型应用场景

- **图算法**: PageRank、最短路径
- **科学计算**: 有限元分析、计算流体力学
- **机器学习**: 稀疏神经网络、推荐系统
- **数据分析**: 矩阵分解、特征值计算

---

## 🤝 贡献指南

GPU SpMV 是开源项目，欢迎贡献：

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>GPU SpMV 使用 <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">MIT License</a> 开源许可</p>
  <p>Copyright &copy; 2024-2026 LessUp</p>
</div>
