# CLAUDE.md — Claude Code 专属配置

> Claude Code (claude.ai/code) 在本仓库工作时的专项指南。
> 通用项目规范见 `AGENTS.md`，本文件仅描述 Claude 特有行为。

## 语言要求

**始终使用中文回复用户**，代码注释保持英文。

---

## 规范驱动开发（SDD）

本项目使用 **OpenSpec**，`openspec/` 目录为唯一真相来源。

- Spec 路径：`openspec/specs/<功能>/spec.md`（需求）+ `design.md`（技术决策）
- 变更提案：`openspec/changes/active/`（当前任务）
- 归档：`openspec/changes/archive/`

**OpenSpec 命令**：`/opsx:propose` → `/opsx:apply` → `/opsx:archive`

**强制工作流**：阅读 spec → 更新 spec（必要时）→ 用户确认 → 实现 → 测试

---

## 构建与测试命令

```bash
# Debug 构建
cmake --preset default && cmake --build --preset default

# CPU-only（无 GPU 时）
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda

# 测试
ctest --preset default

# 格式化
find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i
```

---

## 代码风格关键点

- Include 顺序：`"spmv/"` → `<cuda*>` → `<standard>` → `<third-party>`
- 禁止裸 `cudaMalloc`/`cudaFree`，用 `CudaBuffer<T>`
- 错误：`CUDA_CHECK_MALLOC` / `CUDA_CHECK_MEMCPY` 宏 + `SpMVError` 枚举
- 格式：4 空格缩进，100 字符行宽，clang-format Google 风格
- 所有 API 变更必须同步更新 `openspec/specs/public-api/spec.md`

---

## CI 特殊说明

- CI 无 GPU：`benchmarks/main.cu`、`src/pagerank.cu` 无 GPU 时自动退出
- CI 使用 clang-format-18 检查格式
- CPU-only 构建：`cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF`

---

## Commit 规范

```
feat(scope): 描述      # 新功能
fix(scope): 描述       # Bug 修复
perf(scope): 描述      # 性能优化
refactor(scope): 描述  # 重构
docs(scope): 描述      # 文档
test(scope): 描述      # 测试
ci(scope): 描述        # CI/CD
```
