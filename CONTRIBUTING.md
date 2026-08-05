# 贡献指南

贡献保持小、可验证、聚焦核心 SpMV 库。

## 开发环境

```bash
git clone https://github.com/AICL-Lab/gpu-spmv.git
cd gpu-spmv

cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

无 GPU 环境可使用：

```bash
cmake --preset cpu-only
cmake --build --preset cpu-only
ctest --preset cpu-only
```

Release 构建：

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## GPU CI

内核正确性测试（`tests/test_spmv.cu`、`tests/test_bandwidth.cu`）需要真实
NVIDIA GPU，GitHub 托管 runner 不提供。CI 因此有两道 CUDA 门：

- `cuda-compile`：始终运行，在 NVIDIA 容器里编译库和测试。
- `cuda-test`：在 self-hosted GPU runner 上跑完整测试套件。仅当仓库变量
  `CUDA_RUNNER_LABEL` 被设置时才触发（Settings → Secrets and variables →
  Actions → Variables），把它设为装有 CUDA toolkit、CMake、C++ 编译器的
  self-hosted runner 标签。

合并内核改动前，请在 GPU 机器上本地跑一遍：

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## 示例与基准

- `examples/basic_spmv.cpp` 默认构建（`SPMV_BUILD_EXAMPLES=ON`），CUDA 和
  CPU-only 构建均可运行。
- 合成基准工具需 `-DSPMV_BUILD_BENCHMARKS=ON`（仅 CUDA），也可加载 Matrix
  Market 文件：`./build/spmv_bench matrix.mtx`。

## 什么属于这个仓库

好的贡献：

- 改进 CSR / ELL 存储或校验
- 改进内核选择或执行可靠性
- 修复正确性、内存安全或错误报告问题
- 精简核心库文档

不好的贡献：

- 新增 AI 治理层或仓库专用 agent 工作流
- 不属于核心 SpMV 库的展示型模块
- 增加的维护成本大于价值的大型流程框架

## 代码规范

- 使用 C++17
- 4 空格缩进，100 字符行宽
- 优先复用现有 helper 和显式错误处理
- 不要引入裸 `cudaMalloc` / `cudaFree`，使用 `CudaBuffer<T>`
- include 顺序：项目 → CUDA → 标准库 → 第三方

格式化改动文件：

```bash
find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  | xargs clang-format -i
```

## 提交前检查

1. 跑相关构建和测试命令。
2. 用户可见行为变化时更新 README / docs。
3. 保持改动聚焦，不要捆绑无关清理。
4. 需要时在根目录 `CHANGELOG.md` 记录项目级变更。

## Commit 信息

使用 Conventional Commits：

```text
feat(scope): 描述
fix(scope): 描述
refactor(scope): 描述
docs(scope): 描述
test(scope): 描述
```
