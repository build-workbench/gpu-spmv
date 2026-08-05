<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B" alt="C++">
  <img src="https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/平台-Linux%20%7C%20Windows-blue" alt="Platform">
</p>

<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>聚焦核心能力的 CUDA 稀疏矩阵向量乘法库</strong>
</p>

<p align="center">
  <em>CSR + ELL 格式 · 4 种内核 · 显式错误处理 · 更小维护面</em>
</p>

<p align="center">
  <a href="https://github.com/AICL-Lab/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/AICL-Lab/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://aicl-lab.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/文档-GitHub%20Pages-2EA44F?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/AICL-Lab/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/许可证-MIT-green" alt="License">
  </a>
</p>

## 项目定位

GPU SpMV 是一个 C++17 / CUDA 稀疏矩阵向量乘法库，仓库现在只保留核心库本身：

- **存储层**：CSR 与 ELL 两种稀疏格式
- **执行层**：Scalar CSR、Vector CSR、Merge Path、ELL Kernel
- **工程约束**：`CudaBuffer<T>` RAII、显式 `SpMVError`、CPU 参考路径、聚焦测试

展示型模块和 AI 治理框架已经移除，目标是让代码库更小、更直接、更容易维护。

## 快速开始

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

`cuda-linux` preset 会固定系统 GCC/G++ 作为 host compiler，避免 Conda
编译器串进 nvcc 链路。

## 最小示例

完整可编译版本见 [`examples/basic_spmv.cpp`](examples/basic_spmv.cpp)
（默认随构建编译；无 CUDA 构建时自动退化为 CPU 路径）：

```cpp
#include <spmv/csr_matrix.h>
#include <spmv/cuda_buffer.h>
#include <spmv/spmv.h>

int main() {
    float dense[] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 4.0f,
        0.0f, 0.0f, 5.0f,
    };

    spmv::CSRMatrix* csr = spmv::csr_create(3, 3, 5);
    spmv::csr_from_dense(csr, dense, 3, 3);
    spmv::csr_to_gpu(csr);

    spmv::CudaBuffer<float> d_x(3);
    spmv::CudaBuffer<float> d_y(3);
    const float h_x[] = {1.0f, 1.0f, 1.0f};
    d_x.copyFromHost(h_x, 3);

    spmv::SpMVConfig config = spmv::spmv_auto_config(csr);
    spmv::SpMVResult result = spmv::spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
    spmv::csr_destroy(csr);

    return result.error_code == 0 ? 0 : 1;
}
```

默认情况下 `spmv_csr` 会阻塞直到 `d_y` 完成并报告计时指标。
设置 `config.enable_timing = false` 可只把 kernel 入队到 stream
（不创建 event、不同步），便于流水线编排；读取 `d_y` 前需自行同步。

## 目录结构

```text
gpu-spmv/
├── include/spmv/   # 公共头文件
├── src/            # 核心库实现
├── tests/          # 单元测试与回归测试
├── examples/       # 最小可运行示例
├── tools/          # 基准测试工具（SPMV_BUILD_BENCHMARKS=ON）
├── docs/           # GitHub Pages 文档站
├── CHANGELOG.md    # 唯一更新日志
└── CMakeLists.txt
```

## 文档导航

文档站地址：**https://aicl-lab.github.io/gpu-spmv/**。

| 页面 | 用途 |
|:-----|:-----|
| [快速开始](https://aicl-lab.github.io/gpu-spmv/zh/quickstart) | 安装与构建流程 |
| [API 参考](https://aicl-lab.github.io/gpu-spmv/zh/api/spmv) | 核心公开接口 |
| [架构概览](https://aicl-lab.github.io/gpu-spmv/zh/architecture/overview) | 数据流与内核选择 |
| [性能优化](https://aicl-lab.github.io/gpu-spmv/zh/performance/optimization-guide) | 实用调优建议 |
| [示例代码](https://aicl-lab.github.io/gpu-spmv/zh/examples/basic-spmv) | 端到端用法 |

版本历史只保留在根目录 [CHANGELOG.md](CHANGELOG.md)。

## 参与贡献

贡献流程保持简单：

1. 只做能改善核心库的变更。
2. 保持 RAII 资源管理，不要引入裸 `cudaMalloc` / `cudaFree`。
3. 运行现有构建和测试命令。
4. 行为变化时同步更新相关文档。

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

MIT 许可证，详见 [LICENSE](LICENSE)。
