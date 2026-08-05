# 更新日志

本项目所有值得记录的变更都写在这个文件里。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added
- `SpMVConfig::enable_timing`：设为 `false` 可在不创建 CUDA event、不同步的情况下把 kernel 入队到 stream，便于异步流水线编排（默认 `true` 保留阻塞+计时行为）。
- `spmv_result_error()` 类型化访问器，以及 `spmv_auto_config_ell()` 提供 API 对称性。
- `csr_read_matrix_market()`（`spmv/market_io.h`）：Matrix Market 坐标格式读取器（real/integer/pattern、general/symmetric，重复项求和）。
- 可运行的 `examples/basic_spmv.cpp`（默认构建，CPU-only 构建也可用）和合成基准工具 `tools/spmv_bench.cu`（`SPMV_BUILD_BENCHMARKS=ON`）。
- CI `cuda-test` job：当仓库变量 `CUDA_RUNNER_LABEL` 被设置时，在 self-hosted runner 上跑完整 GPU 测试套件。
- 序列化格式版本 2：完整性校验现在也覆盖 `values` 数组。版本 1 文件仍可读取。

### Fixed
- `csr_read_matrix_market` 不再在头部声明条目数超过文件实际内容时崩溃（`std::length_error`）；这类文件以 `FILE_IO` 拒绝，条目数上限为 `INT_MAX / 2`（对称展开上界）。
- 安装包现在正确导出 include 目录：`include(GNUInstallDirs)` 之前在 target 定义之后执行，导致 `gpu_spmv::spmv` 丢失 `INTERFACE_INCLUDE_DIRECTORIES`，`find_package()` 消费者无法编译。
- 创建和转换函数（`csr_create`、`csr_from_dense`、`ell_create`、`ell_from_dense`、`ell_from_csr`）改用 nothrow 分配，失败时返回文档所述的 `nullptr` / `OUT_OF_MEMORY`，不再让 `std::bad_alloc` 逃出 C 风格 API。
- `csr_create` / `csr_from_dense` 拒绝 `rows == INT_MAX`，避免 `rows + 1` 分配溢出。
- `ell_from_csr` 先分配再释放旧数组，分配失败时矩阵保持完整（与 `ell_from_dense` / `csr_from_dense` 一致）。
- `CudaBuffer::resize` 在 device-to-device 拷贝失败时保留原 buffer，不再留下 null 指针 + 旧 size。
- Kernel grid 计算（scalar / vector / merge path / ELL）使用溢出安全的 ceil division，应对极端矩阵规模。
- README 最小示例现在可编译（`CudaBuffer` 暴露 `get()`，不是 `data()`）。
- `SpMVResult::bandwidth_gb_s` 改为从 `elapsed_ms` 推导，不再用独立 host 计时器，结果字段相互一致。
- `spmv_csr`/`spmv_ell` 零工作量路径（空矩阵 / 零长向量）在开启计时时也像 kernel 路径一样同步，阻塞行为不再依赖矩阵内容。
- Merge Path grid 改为按 `nnz` 而非 `num_rows` 划分，恢复该内核存在意义的不规则矩阵并行度。
- ELL kernel 使用 64 位索引运算，`num_rows * max_nnz_per_row > INT_MAX` 不再溢出。
- `csr_deserialize`/`ell_deserialize` 拒绝头部声明负载超过文件实际大小的情形（在分配之前），并保护 `rows + 1` 分配免受整数溢出。
- `csr_get_element` 对列索引未排序的 CSR 矩阵返回正确值。
- L2 persisting access-policy hint 在退出时恢复，`spmv_csr`/`spmv_ell` 不再对调用方 stream 留下副作用。

### Changed
- 仓库缩减为核心 CSR / ELL SpMV 库，移除仓库专用 AI 治理文件。
- 贡献流程、GitHub 模板、GitHub Pages 内容简化为匹配更小的核心范围。
- 新增基于系统 GCC/G++ 的 Linux CUDA preset，Conda host compiler 快速失败指引。
- `spmv_cpu_csr`/`spmv_cpu_ell` 返回 `int` 错误码，不再静默忽略非法输入。
- CI `build-cpu` job 现在安装包并构建 `find_package()` 消费者作为打包冒烟测试。
- `CudaBuffer::resize` 保留已有元素（类似 `std::vector::resize`），不再丢弃。
- 内部 `select_kernel()` 不再接受未使用的 `num_cols` 参数。

### Removed
- OpenSpec 规范、Claude / Copilot 仓库指令文件、本地 skill 配置。
- 内置 PageRank 和基准模块及其测试、文档页面。
- GitHub Pages 镜像 changelog；根目录 `CHANGELOG.md` 是唯一 changelog。
- 英文 README 和英文文档站（`docs/en/`），项目文档统一为中文。

## [1.0.0] - 2025-04-16

首个稳定版本。

### Added
- CSR / ELL 两种稀疏矩阵格式，完整操作（创建、转换、序列化、校验、统计）。
- 四种 CUDA 内核：Scalar CSR、Vector CSR、Merge Path、ELL Kernel。
- 基于矩阵统计（avg_nnz、skewness）的自动内核选择。
- RAII 资源管理：`CudaBuffer<T>`、显式 `SpMVError` 错误码、CPU 参考路径。
- CMake Presets、CPU-only 配置选项、Google Test 测试套件、GitHub Actions CI。

### Security
- 尺寸计算的整数溢出保护。
- 矩阵操作的内存边界检查。

## [0.1.0] - 2025-03-01

初始原型：基本项目结构、CSR 矩阵实现、简单 SpMV GPU 内核、CMake 构建配置。

---

[1.0.0]: https://github.com/AICL-Lab/gpu-spmv/releases/tag/v1.0.0
[0.1.0]: https://github.com/AICL-Lab/gpu-spmv/tree/7d6dd0c
