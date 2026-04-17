# Workflow CPU-safe CI 调整

**日期**: 2026-03-13  
**关联 Issue**: #3  
**类型**: 基础设施

---

## 🔄 变更 (Changed)

### CI Workflow 重构
- 将 `.github/workflows/ci.yml` 从 GitHub Hosted Runner + CUDA 容器构建
- 调整为仅保留 CPU-safe 的 `Format Check` 任务

### 触发器配置
- 补回 `push`、`pull_request` 与 `workflow_dispatch` 触发
- 避免仅有手动触发却长期红灯的问题

### 格式检查
- 统一使用 `jidicula/clang-format-action` 执行格式校验
- 排除 `build`、`third_party`、`external`、`vendor` 等目录

## 🐛 修复 (Fixed)

### CI 稳定性
- 修复 GitHub Hosted Runner 无法使用 GPU 导致的 CI 持续失败
- 修复格式检查覆盖范围不一致的问题

---

## 背景

该仓库原先的 CI 依赖 `nvidia/cuda` 容器，但 GitHub Hosted Runner 不提供可用 GPU，导致工作流长期处于无效或失败状态。

本次调整将主线检查收敛为稳定可通过的静态格式校验。

---

## 影响

### 对开发者
- ✅ CI 状态可见且稳定
- ✅ PR 不再因 CUDA 环境问题被阻塞
- ⚠️ GPU 测试移至本地执行

### 后续计划
- [ ] 评估 GitHub Actions 的 GPU runner 选项
- [ ] 或搭建自托管 GPU runner
