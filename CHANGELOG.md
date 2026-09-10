# 更新日志

GPU SpMV 是一个 C++17 / CUDA 稀疏矩阵向量乘法库，提供 CSR 与 ELL 两种稀疏格式、多种内核
实现与显式错误处理，只保留聚焦核心能力的库本体。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- 异步计时模式：`SpMVConfig::enable_timing = false` 时仅入队内核而不插 CUDA event 与同步，
  使 SpMV 调用可流水线化。
- Matrix Market（NIST）coordinate 文件读取 `csr_read_matrix_market()`，支持 real /
  integer / pattern 与 general / symmetric，重复项求和、对称矩阵展开。
- 可运行示例 `examples/basic_spmv.cpp` 与合成基准工具 `tools/spmv_bench.cu`。
- 序列化格式升级到 v2（完整性校验和同时覆盖 `values` 数组，v1 文件仍可读），并补充
  `spmv_result_error()`、`spmv_auto_config_ell()` 等 API。
- 自托管 runner 上的 `cuda-test` CI 任务，以及 Doxygen 注释、贡献指南与规范驱动开发
  （SDD）工作流文档。

### 变更

- 仓库收敛：移除 VitePress 文档站、CMake 分发包链路（`gpu_spmvConfig.cmake`）、多余 CI /
  治理文件与 GitHub 模板，只保留核心库。
- 架构深化：抽出 `internal/csr_device`、`ell_device`、`kernel_selector` 与 `texture_cache`
  等内部执行缝隙，隐藏设备细节并统一输入校验；删除死代码 `matrix_wrapper.h`。
- 内核现代化：重写 `spmv_kernels.cu`，Merge Path 网格改按 `nnz` 划分以恢复高度倾斜矩阵
  的并行度，ELL 内核改用 64 位索引运算。
- 文档站多次重构：Git Pages 升级为技术白皮书定位（自定义首页、GitHub Dark 配色、交互式
  图表），随后由 Jekyll 迁移到 VitePress，最终随仓库收敛整体移除；README 仅保留中文。
- 项目治理文件与文档中文化，版权持有者统一为 build-workbench，默认分支迁移至 `main`。

### 修复

- 序列化加固：拒绝声称载荷超过文件内容的头部（且在分配前拦截），并对 `rows + 1` 分配做
  整数溢出防护。
- `CudaBuffer::resize` 改为保留已有元素（对齐 `std::vector::resize` 语义）；`csr_get_element`
  对行内列索引未排序的矩阵返回正确值。
- 修正零工作量路径（空矩阵 / 零长度向量）的同步语义，使其与内核路径一致；恢复退出时的 L2
  persisting access-policy hint，避免污染调用方 stream。
- 加固失败路径、打包与溢出防护，显式处理基准与 PageRank 的错误；CUDA 不可用时构建快速失败。

### 移除

- 移除内置 PageRank 与 benchmark 模块及其测试与文档页。
- 移除 OpenSpec 规范、Claude / Copilot 仓库指令文件与本地 skill 配置。

## [v1.0.0] - 2026-04-16

### 新增

- 首个版本：CSR 与 ELL 稀疏矩阵格式、设备端 SpMV 内核、纯 CPU 参考实现与单元测试。
- CMake 构建体系与 CMakePresets、`cuda_buffer` / `cuda_compat` 兼容层、EditorConfig 与
  CI 工作流。
- 基准测试、矩阵测试与带宽测量，以及中英双语 README 和基于 GitHub Pages 的文档站。

### 变更

- SpMV 函数增加向量大小参数，并优化 Phase 1 执行路径。
- 整理 CMake 与公共头文件，移除废弃的 `common.cpp`，完善 CSR / ELL 与内核测试。
- CI 调整为仅 CPU 安全的格式检查（GitHub 托管 runner 无 GPU，禁用 CUDA 构建任务）。

### 修复

- 加固 CUDA 构建回退路径（CUDA 不可用时快速失败）并补齐 SpMV 测试覆盖。
