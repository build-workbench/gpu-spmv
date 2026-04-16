---
layout: default
title: 更新日志
lang: zh
---

<p align="right">
  <a href="changelog.en.html">🇺🇸 English</a>
</p>

# 📋 更新日志

本文档记录 GPU SpMV 项目的所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [1.0.0] - 2025-04-16

### 🎉 首个稳定版本

#### ✨ 新增功能

**核心功能**
- CSR (Compressed Sparse Row) 稀疏矩阵格式完整支持
- ELL (ELLPACK) 稀疏矩阵格式完整支持
- 四种 CUDA Kernel: Scalar CSR, Vector CSR, Merge Path, ELL
- 基于矩阵统计特征的自动 Kernel 选择
- 纹理缓存支持，执行上下文复用

**性能与测试**
- 带宽度量计算
- 基准测试框架
- GPU vs CPU 性能对比

**应用示例**
- PageRank 图算法实现

**工程质量**
- RAII 资源管理 (`CudaBuffer`, `CudaTimer`, `SpMVExecutionContext`)
- 语义化错误码系统
- 完整的 Google Test 测试套件
- CMake Presets 构建配置
- GitHub Actions CI/CD

#### 🔒 安全性
- 整数溢出保护
- 内存安全检查

#### 🚀 性能优化
- ELL Column-major 合并访存
- Warp 级 Shuffle 归约
- 负载均衡优化 (Merge Path)

---

## [0.1.0] - 2025-03-01

### 🚀 初始版本

- 项目基础结构
- 基础 CSR 矩阵实现
- 简单 SpMV GPU Kernel
- CMake 构建配置

---

## 版本历史

| 版本 | 日期 | 状态 | 亮点 |
|:----:|:----:|:----:|:-----|
| 1.0.0 | 2025-04-16 | 稳定 | 首个稳定版本，完整功能 |
| 0.1.0 | 2025-03-01 | 归档 | 初始原型 |

---

## 迁移指南

### 升级到 1.0.0

从预发布版本升级无破坏性变更。

#### 推荐更新

1. **使用命名常量**
   ```cpp
   // 旧
   config.block_size = 256;
   
   // 新
   config.block_size = spmv::DEFAULT_BLOCK_SIZE;
   ```

2. **使用执行上下文复用**
   ```cpp
   // 旧
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols);
   }
   
   // 新
   SpMVExecutionContext context;
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols, &context);
   }
   ```

---

## 未来路线

### 计划 [1.1.0]
- [ ] COO 格式支持
- [ ] 混合 CSR/ELL 格式
- [ ] 多 GPU 支持
- [ ] 批量 SpMV 操作

### 考虑中
- [ ] BFloat16 精度支持
- [ ] 自动格式选择调优
- [ ] Python 绑定

---

<div align="center">

**[← 性能优化](performance)** · **[ 🏠 首页 →](index)**

</div>
