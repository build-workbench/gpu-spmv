#AGENTS.md — GPU SpMV AI Agent Guidelines

> 面向所有 AI 编码助手（GitHub Copilot、Claude、Codex）的项目工作规范。
> **始终使用中文回复用户。**

---

## 项目速览

**GPU SpMV** — 基于 CUDA 的高性能稀疏矩阵向量乘法库（C++17）。

| 要素 | 详情 |
|------|------|
| 语言 | C++17 + CUDA C++ |
| 构建 | CMake 3.18+，presets（无 Makefile） |
| 测试 | Google Test，property tests ≥ 100 次迭代 |
| 格式化 | clang-format-18（Google 风格，CI 强制） |
| GPU | Compute Capability 7.0+（Volta 以上） |
| SDD | `openspec/` 是唯一真相来源 |

核心组件：4 种 CUDA Kernel（Scalar CSR / Vector CSR / Merge Path / ELL）+ CSR/ELL 两种稀疏格式 + 自动 Kernel 选择 + PageRank 算法示例。

---

## 开发工作流（MANDATORY）

### OpenSpec 驱动开发

`openspec/` 是唯一真相来源，所有功能开发必须遵循：

```
openspec/
├── config.yaml          # 项目配置与规则
├── specs/               # 各功能规范（唯一真相来源）
│   ├── csr-format/      ├─ spec.md + design.md
│   ├── ell-format/      ├─ spec.md + design.md
│   ├── spmv-kernels/    ├─ spec.md + design.md
│   ├── public-api/      ├─ spec.md（所有 API 变更必须同步更新）
│   ├── error-handling/  ├─ spec.md
│   ├── benchmark/       ├─ spec.md
│   ├── pagerank/        └─ spec.md
│   └── property-tests/  └─ spec.md（测试要求）
└── changes/
    ├── active/          # 当前迭代任务（从这里取任务）
    └── archive/         # 已完成变更
```

### AI 必须遵循的 4 步流程

1. **阅读 Spec（必须）**：先读 `openspec/specs/<功能>/spec.md` 和 `design.md`
2. **更新 Spec（新功能必须）**：用 `/opsx:propose` 创建变更提案，等用户确认
3. **实现（100% 遵循 spec）**：API 必须匹配 `openspec/specs/public-api/spec.md`
4. **测试验证（必须）**：基于 `openspec/specs/property-tests/spec.md` 写测试

### 分支策略

- **小改动**（< 200 行）：直接提交到 `master`
- **大功能**：使用短寿命分支 `feat/xxx`，完成后立即合并

---

## 构建与测试

```bash
#Debug 构建
cmake --preset default && cmake --build --preset default

#CPU - only（无 GPU 环境，CI 使用此配置）
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF && cmake --build build-no-cuda

#测试
ctest --preset default

#格式化
find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i
```

> **CI 无 GPU**：需要 CUDA 设备的测试在 CI 中会跳过。

---

## 代码规范

### Include 顺序（严格遵循）

```cpp
#include "spmv/xxx.h"  // 1. 项目头文件

#include <cuda_runtime.h>  // 2. CUDA 头文件

#include <gtest/gtest.h>  // 4. 第三方库
#include <vector>         // 3. 标准库
```

### 命名约定

| 类别 | 风格 | 示例 |
|------|------|------|
| 类型/结构体 | PascalCase | `CSRMatrix`, `SpMVConfig` |
| 函数 | snake_case | `csr_create`, `spmv_csr` |
| 常量/枚举 | UPPER_SNAKE_CASE | `WARP_SIZE`, `SCALAR_CSR` |
| 命名空间 | lowercase | `spmv::` |
| 私有成员 | snake_case + 后缀 `_` | `ptr_`, `size_` |

### 核心约束

- **禁止**裸 `cudaMalloc`/`cudaFree`，必须用 `CudaBuffer<T>` 或 RAII wrapper
- 错误处理：`CUDA_CHECK_MALLOC`/`CUDA_CHECK_MEMCPY` 宏，返回 `SpMVError` 枚举
- 格式：4 空格缩进，100 字符行宽

### Kernel 选择逻辑

```
avg_nnz_per_row < 4              → Scalar CSR（1 线程/行）
avg_nnz_per_row ≥ 4 且 skewness < 10  → Vector CSR（1 warp/行）
avg_nnz_per_row ≥ 4 且 skewness ≥ 10  → Merge Path（完美负载均衡）
ELL format                        → ELL Kernel（合并访存）
```

---

## Commit 规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <描述>

类型: feat | fix | perf | refactor | test | docs | build | ci | chore
示例: fix(csr): 修复空行元素查找越界问题
```

---

## 开发工具配置

- **LSP**：`.clangd`（clangd 配置，适用所有编辑器）
- **Git Hooks**：`git config core.hooksPath .githooks`（激活 pre-commit 格式检查）
- **文档**：https://lessup.github.io/gpu-spmv/
