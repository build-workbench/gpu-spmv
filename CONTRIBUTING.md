# Contributing to GPU SpMV

本项目处于**稳定维护阶段**，主要接受 bug 修复和文档改进。

---

## Bug 报告

提交 Issue 时请包含：
- 复现步骤
- 预期 vs 实际行为
- 环境信息（OS、CUDA 版本、GPU 型号）

---

## 提交代码

1. Fork 仓库，创建短寿命分支 `fix/xxx` 或 `feat/xxx`
2. 遵循 OpenSpec 工作流（见 `AGENTS.md`）：先阅读 `openspec/specs/` 中相关规范
3. 格式化代码：`find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i`
4. 确保 CPU-only 构建通过：`cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda`
5. 提交遵循 [Conventional Commits](https://www.conventionalcommits.org/)：`fix(csr): 描述`
6. 发起 PR，立即合并，不堆积分支

---

## 本地 Git Hooks（推荐）

激活自动 clang-format 检查：
```bash
git config core.hooksPath .githooks
```

---

## 文档

- 在线文档：https://lessup.github.io/gpu-spmv/
- API 规范：`openspec/specs/public-api/spec.md`
