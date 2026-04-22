---
layout: default
title: 更新日志
nav_order: 7
permalink: /changelog
lang: zh
---

<p align="right">
  <a href="changelog.en">🇺🇸 English</a>
</p>

# 更新日志
{: .no_toc }

GPU SpMV 的版本历史和更新记录。
{: .fs-6 .fw-300 }

---

## v1.1.0 (2026-04-22)

### 新增

- 🎨 **全新 GitHub Pages 站点** - 重新设计的文档网站
- 🚀 **性能可视化** - 添加性能对比图表
- 📚 **双语文档** - 完整的中英文文档支持

### 优化

- 重构首页设计，突出产品特性
- 统一中英版本内容深度
- 优化导航结构

---

## v1.0.1 (2026-03-22)

### 优化

- **Kernel 选择优化** - 改进自动选择算法的准确性
- **内存管理** - 优化 CudaBuffer 的移动语义
- **编译速度** - 改进 CMake 配置，加快构建

### 修复

- 修复 ELL 格式转换时的边界检查问题
- 修复 PageRank 在特定收敛条件下的数值稳定性

---

## v1.0.0 (2026-03-10)

### 正式发布

- 🎉 **首个稳定版本**
- 4 种优化 Kernel（Scalar/Vector/Merge Path/ELL）
- CSR 和 ELL 双格式支持
- 完整的 RAII 资源管理
- 100+ 测试用例覆盖
- PageRank 算法实现

### 特性

- 智能 Kernel 自动选择
- 高达 70%+ 带宽利用率
- 跨平台支持（Linux/Windows）
- CMake Presets 一键构建

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>完整发布说明见 <a href="https://github.com/LessUp/gpu-spmv/releases">GitHub Releases</a></p>
</div>
