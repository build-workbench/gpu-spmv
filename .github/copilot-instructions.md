# GitHub Copilot Instructions — GPU SpMV

## 语言要求

**始终使用中文回复用户**。代码注释、变量名、commit message 保持英文。

---

## 项目背景

GPU SpMV 是基于 CUDA 的高性能稀疏矩阵向量乘法库（C++17）。

- 4 种 CUDA Kernel：Scalar CSR / Vector CSR / Merge Path / ELL
- 2 种稀疏格式：CSR（通用）/ ELL（均匀行长）
- 自动 Kernel 选择：基于 avg_nnz_per_row 和 skewness 指标
- PageRank 算法示例

---

## 规范驱动开发（SDD）

`openspec/` 是唯一真相来源，所有功能开发必须遵循：

1. 先读 `openspec/specs/<功能>/spec.md`（需求）和 `design.md`（设计决策）
2. 变更提案：`/opsx:propose` → 确认 → `/opsx:apply` → `/review` → `/opsx:archive`
3. 所有 API 变更必须同步更新 `openspec/specs/public-api/spec.md`
4. 当前待办任务：见 `openspec/changes/active/`

---

## 代码规范

### Include 顺序（严格）
```cpp
#include "spmv/xxx.h"     // 1. 项目头文件
#include <cuda_runtime.h>  // 2. CUDA
#include <vector>           // 3. 标准库
#include <gtest/gtest.h>    // 4. 第三方
```

### 核心约束
- **禁止**裸 `cudaMalloc`/`cudaFree`，必须用 `CudaBuffer<T>` 或 RAII wrapper
- 错误处理：`CUDA_CHECK_MALLOC`/`CUDA_CHECK_MEMCPY` 宏，返回 `SpMVError` 枚举
- 格式：4 空格缩进，100 字符行宽，clang-format Google 风格
- Property tests 必须 ≥ 100 次迭代随机矩阵

### 命名
| 类别 | 风格 | 示例 |
|------|------|------|
| 类型/结构体 | PascalCase | `CSRMatrix`, `SpMVConfig` |
| 函数 | snake_case | `csr_create`, `spmv_csr` |
| 常量/枚举值 | UPPER_SNAKE_CASE | `WARP_SIZE`, `SCALAR_CSR` |
| 私有成员 | snake_case + `_` | `ptr_`, `size_` |

---

## 构建命令

```bash
cmake --preset default && cmake --build --preset default  # Debug
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda  # CPU-only
ctest --preset default  # 测试
```

**CI 无 GPU**：`benchmarks/main.cu` 和 `src/pagerank.cu` 在无 GPU 时自动退出。

---

## Commit 规范

```
feat(scope): 简短描述（中文可用于描述部分）
fix(csr): 修复空行元素查找越界
perf(ell): 优化列主序访存模式
```

---

## 开发工具配置

- **LSP**：`.clangd`（clangd 配置，适用所有编辑器）
- **Git Hooks**：`git config core.hooksPath .githooks`（激活 pre-commit 格式检查）
- **Spec 工具**：OpenSpec 命令（`/opsx:propose` 等）
- **文档**：https://lessup.github.io/gpu-spmv/
