# 项目收尾完善提案

**提案 ID**: project-completion
**状态**: 🚧 Active
**创建日期**: 2026-04-24
**优先级**: 高（项目收尾，完善后归档）

---

## 背景

GPU SpMV 库核心功能已完成（v1.0.0），现进入**收尾完善阶段**。
目标：修复所有已知问题，完善质量保证，完成后归档项目。

---

## 收尾任务清单

### T1: 代码质量修复

**T1-1: 修复 clang-tidy 静态分析警告**
- 运行：`cmake -S . -B build -DSPMV_REQUIRE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && clang-tidy -p build src/*.cpp include/spmv/*.h`
- 修复所有 `modernize-use-override`、`modernize-use-nullptr` 等警告
- 文件范围：`src/*.cpp`, `include/spmv/*.h`

**T1-2: 验证 CPU-only 构建无警告**
- 命令：`cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda 2>&1 | grep -E "warning|error"`
- 修复所有编译警告（`-Wall -Wextra` 级别）

**T1-3: 确认 property tests ≥ 100 次迭代**
- 检查 `tests/test_spmv.cu`、`tests/test_csr.cpp`、`tests/test_ell.cpp` 中的 property tests
- 若少于 100 次迭代，调整到 100 次

### T2: 文档完善

**T2-1: 更新 openspec specs 与实现对齐**
- 检查 `openspec/specs/public-api/spec.md` 是否与 `include/spmv/spmv.h` 实际 API 完全对齐
- 更新 `openspec/specs/spmv-kernels/spec.md` 中的 Kernel 选择阈值（确认 avg_nnz < 4 和 skewness < 10）

**T2-2: README 最终检查**
- 确认 README.md 和 README.zh-CN.md 中所有代码示例可以实际编译运行
- 确认 GitHub Pages 链接有效

**T2-3: CHANGELOG.md 补充当前版本状态**
- 版本状态：v1.0.0 稳定，已知修复记录到 v1.0.1（若有 bug 修复）

### T3: 测试覆盖补充

**T3-1: 验证 ELL 格式边界情况**
- 测试：空矩阵、单行矩阵、max_nnz_per_row = 1 的极端情况
- 文件：`tests/test_ell.cpp`

**T3-2: 验证 Merge Path Kernel 空行处理**
- 测试：含大量空行（0 个非零元素）的高度稀疏矩阵
- 文件：`tests/test_spmv.cu`

**T3-3: PageRank 收敛性测试**
- 测试：确认在标准图（如 Karate Club 图）上 PageRank 收敛到已知值
- 文件：`tests/test_pagerank.cu`

### T4: 工程化完善（可选）

**T4-1: 添加 GitHub Release v1.0.0**
- 使用 `gh release create v1.0.0 --title "GPU SpMV v1.0.0" --notes-file CHANGELOG.md`
- 仅在 T1/T2/T3 全部完成后执行

**T4-2: 为 CPU-only 测试添加 GTest 执行**
- 当前 CPU-only 构建不包含测试目标，考虑添加纯 CPU 单元测试（不需要 GPU）
- 参考：`tests/test_common.cpp`，`tests/test_csr.cpp` 中的 CPU-side 逻辑

---

## 验收标准

- [ ] `cmake --preset default && cmake --build --preset default` 无错误
- [ ] `cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda` 无错误无警告
- [ ] clang-format 检查通过：`find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format --dry-run --Werror`
- [ ] `openspec/specs/public-api/spec.md` 与实际 API 对齐
- [ ] `CHANGELOG.md` 准确反映 v1.0.0 状态

---

## 完成后操作

```bash
# 全部任务完成后执行
/opsx:archive
# 或手动移动：
# mv openspec/changes/active/project-completion openspec/changes/archive/2026-04-project-completion
```
