---
layout: default
title: 更新日志
---

# 更新日志

本文档记录项目的所有重要变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/)，版本号遵循 [语义化版本](https://semver.org/)。

---

## 目录

- [v1.0.0](#v100---2025-04-16)
- [v0.1.0](#v010---2025-03-01)
- [迁移指南](#迁移指南)
- [未来路线图](#未来路线图)

---

## [v1.0.0] - 2025-04-16

### 新增功能

#### 核心功能

- **CSR (Compressed Sparse Row) 稀疏矩阵格式**
  - 创建、销毁、序列化、反序列化操作
  - 稠密矩阵转换和元素访问
  - GPU 内存管理与自动同步
  - 统计信息计算（平均/最大/最小每行非零元素数、偏斜度）

- **ELL (ELLPACK) 稀疏矩阵格式**
  - Column-major 存储，优化 GPU 内存合并访问
  - CSR 到 ELL 格式转换
  - 填充检测和存储效率度量

- **四种优化的 CUDA Kernel**
  - **Scalar CSR**: 一个线程处理一行，适合极稀疏矩阵
  - **Vector CSR**: 一个 Warp (32线程) 处理一行，适合均匀分布
  - **Merge Path**: 工作量完全均匀分配，适合高度不均匀矩阵
  - **ELL Kernel**: Column-major 访问，适合均匀行长度

- **自动 Kernel 选择**
  - 矩阵统计特征分析
  - 基于决策树的 Kernel 推荐
  - 通过 `SpMVConfig` 可配置

- **纹理缓存支持**
  - 可选纹理缓存读取输入向量
  - `SpMVExecutionContext` 复用纹理对象
  - 自动阈值启用 (num_cols > 10000)

#### 性能与基准测试

- **带宽度量**
  - 通过 `std::call_once` 缓存峰值带宽
  - 实际带宽计算
  - 效率百分比报告

- **基准测试框架**
  - 可配置预热和运行次数
  - 统计分析（平均、最小、最大、标准差）
  - GPU vs CPU 对比及加速比
  - JSON 结果导出

#### 应用示例

- **PageRank 算法**
  - GPU 加速迭代计算
  - 可配置阻尼系数和收敛阈值
  - Top-K 节点排名提取
  - 收敛跟踪

#### 工程质量

- **RAII 资源管理**
  - `CudaBuffer<T>`: GPU 内存管理 (memset/fill/resize)
  - `CudaTimer`: 基于 CUDA Events 的精确计时
  - `ScopedTexture`: 纹理对象生命周期管理

- **语义化错误码**
  - `SpMVError` 枚举，描述性错误值
  - `CUDA_CHECK_MALLOC` / `CUDA_CHECK_MEMCPY` 宏
  - `spmv_error_string()` 人类可读错误信息
  - `CudaException` 用于 RAII 错误传播

- **构建系统**
  - CMake Presets (default, release, minimal)
  - CPU-only 配置选项 (`DSPMV_REQUIRE_CUDA=OFF`)
  - 项目版本元数据 (1.0.0)
  - 库和头文件安装规则

- **代码质量**
  - `.clang-format` (Google 风格，4空格缩进，100列)
  - `.editorconfig` 编辑器一致性
  - 跨平台测试路径处理

- **测试**
  - 属性测试，100 次随机迭代
  - CSR/ELL 转换正确性
  - SpMV 计算验证（与 CPU 参考对比）
  - 维度验证测试
  - Kernel 选择器有效性
  - 带宽度量验证
  - PageRank 不变量检查

- **文档**
  - GitHub Pages 站点 https://lessup.github.io/gpu-spmv/
  - 完整 API 参考
  - 性能优化指南
  - 代码示例集合
  - 双语 README

- **CI/CD**
  - GitHub Actions 工作流
  - clang-format 验证
  - CPU-only 构建验证

### 安全

- **整数溢出保护**
  - CSR 操作中的大小计算验证
  - ELL 操作中的大小计算验证
  - 防止极端输入导致的未定义行为

### 性能

- **内存带宽优化**
  - ELL 格式的 Column-major 存储
  - Warp 级 Shuffle 归约（无 bank conflict）
  - 大输入向量的纹理缓存

- **负载均衡**
  - 高度偏斜矩阵的 Merge Path 算法
  - 基于矩阵结构的自动 Kernel 选择

---

## [v0.1.0] - 2025-03-01

### 新增功能

- 初始项目结构
- 基础 CSR 矩阵实现
- 简单 SpMV GPU Kernel
- CMake 构建配置

---

## 版本历史摘要

| 版本 | 日期 | 亮点 |
|------|------|------|
| 1.0.0 | 2025-04-16 | 首个稳定版本，完整功能集 |
| 0.1.0 | 2025-03-01 | 初始原型 |

---

## 迁移指南

### 升级到 1.0.0

从预发布版本升级无破坏性变更。API 现已稳定。

#### 推荐更新

1. **使用命名常量** 替代魔法数字：
   ```cpp
   // 旧
   config.block_size = 256;
   config.use_texture = (cols > 10000);

   // 新（推荐）
   config.block_size = spmv::DEFAULT_BLOCK_SIZE;
   config.use_texture = (cols > spmv::TEXTURE_CACHE_THRESHOLD_COLS);
   ```

2. **使用 `SpMVExecutionContext`** 复用纹理对象：
   ```cpp
   // 旧：每次调用创建/销毁纹理对象
   for (int i = 0; i < iterations; i++) {
       config.use_texture = true;
       spmv_csr(csr, d_x, d_y, &config, cols);
   }

   // 新：跨调用复用纹理对象
   SpMVExecutionContext context;
   config.use_texture = true;
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols, &context);
   }
   ```

3. **一致检查错误码**：
   ```cpp
   SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);
   if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
       fprintf(stderr, "错误: %s\n", spmv_error_string(
           static_cast<SpMVError>(result.error_code)));
   }
   ```

---

## 未来路线图

### 计划中的 1.1.0

- [ ] COO (Coordinate) 格式支持
- [ ] 混合 CSR/ELL 格式
- [ ] 多 GPU 支持
- [ ] 批量 SpMV 操作

### 考虑中

- [ ] BFloat16 精度支持
- [ ] 稀疏矩阵存储格式自动调优
- [ ] 与 cuSPARSE 集成对比
- [ ] Python 绑定

---

[v1.0.0]: https://github.com/LessUp/gpu-spmv/releases/tag/v1.0.0
[v0.1.0]: https://github.com/LessUp/gpu-spmv/tree/7d6dd0c
