---
layout: default
title: 更新日志
parent: 中文文档
nav_order: 6
lang: zh
---

<p align="right">
  <a href="changelog.en">🇺🇸 English</a>
</p>

# 📋 更新日志
{: .no_toc }

项目版本历史和变更记录。
{: .fs-6 .fw-300 }

---

## [1.0.0] - 2025-04-16

### 🎉 首个稳定版本

#### 新增
- CSR 和 ELL 稀疏矩阵格式完整支持
- 四种优化的 CUDA Kernel
- 自动 Kernel 选择
- PageRank 算法实现
- 完整的基准测试框架

#### 优化
- ELL Column-major 合并访存
- Warp 级 Shuffle 归约
- Merge Path 负载均衡

---

## [0.1.0] - 2025-03-01

- 初始项目结构
- 基础 CSR 实现

---

## 查看完整日志

查看 [GitHub Releases](https://github.com/LessUp/gpu-spmv/releases) 获取详细更新信息。
