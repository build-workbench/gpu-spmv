---
layout: default
title: 安装指南
lang: zh
---

<p align="right">
  <a href="installation.en.html">🇺🇸 English</a>
</p>

# 📦 安装指南

本指南将帮助您在系统上安装和配置 GPU SpMV 库。

---

## 目录

- [系统要求](#系统要求)
- [依赖项安装](#依赖项安装)
- [构建步骤](#构建步骤)
- [验证安装](#验证安装)
- [常见问题](#常见问题)
- [卸载](#卸载)

---

## 系统要求

### 必需组件

| 组件 | 最低版本 | 推荐版本 | 说明 |
|:-----|:--------:|:--------:|:-----|
| **CUDA Toolkit** | 11.0 | 12.0+ | NVIDIA GPU 计算必需 |
| **CMake** | 3.18 | 3.25+ | 构建系统 |
| **C++ 编译器** | C++17 | C++17 | GCC 7+ / Clang 5+ / MSVC 2017+ |
| **NVIDIA GPU** | CC 7.0 | CC 8.6+ | Volta 架构或更新 |

### 支持的 GPU 架构

| 架构 | 计算能力 | 代表性 GPU |
|:-----|:--------:|:-----------|
| Volta | 7.0, 7.5 | V100, Titan V |
| Turing | 7.5 | RTX 20 系列, T4 |
| Ampere | 8.0, 8.6 | RTX 30 系列, A100, A6000 |
| Ada Lovelace | 8.9 | RTX 40 系列 |
| Hopper | 9.0 | H100 |

### 可选组件

| 组件 | 用途 |
|:-----|:-----|
| **NVIDIA 驱动** | 450.80.02+ (CUDA 11) / 525.85.12+ (CUDA 12) |
| **Git** | 克隆源代码仓库 |
| **Google Test** | 运行单元测试 (自动下载) |

---

## 依赖项安装

### Ubuntu / Debian

```bash
# 1. 安装 CUDA Toolkit (如未安装)
# 方法 A: 通过 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# 方法 B: 使用现有 CUDA 版本
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# 2. 安装构建工具
sudo apt-get install -y cmake build-essential git

# 3. 验证 CUDA 安装
nvcc --version
nvidia-smi
```

### CentOS / RHEL / Rocky Linux

```bash
# 1. 安装开发工具
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git

# 2. 安装 CUDA (通过 runfile 或 NVIDIA 仓库)
# 参考: https://developer.nvidia.com/cuda-downloads

# 3. 验证安装
nvcc --version
```

### Windows

1. **安装 Visual Studio 2019/2022**
   - 选择"使用 C++ 的桌面开发"工作负载
   - 确保安装了 MSVC v142 或更新版本

2. **安装 CUDA Toolkit**
   - 从 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)下载
   - 选择对应的 Windows 版本

3. **安装 CMake**
   - 从 [CMake 官网](https://cmake.org/download/)下载安装程序
   - 添加到系统 PATH

4. **验证安装**
   ```cmd
   nvcc --version
   cmake --version
   ```

### macOS

> ⚠️ **注意**: macOS 已停止支持 NVIDIA GPU 的 CUDA。请使用 CPU-only 模式进行开发，或切换到 Linux/Windows 环境。

---

## 构建步骤

### 1. 克隆仓库

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
```

### 2. 配置构建 (CMake Presets)

本项目提供预配置的 CMake Presets，推荐使用：

```bash
# Release 构建（推荐用于生产环境）
cmake --preset release

# Debug 构建（用于开发调试）
cmake --preset default

# 仅 CPU 模式（无 GPU 环境）
cmake --preset no-cuda
```

### 3. 执行构建

```bash
# 使用对应 preset 构建
cmake --build --preset release
```

### 4. 自定义构建选项

如需自定义构建参数：

```bash
mkdir build && cd build

# 基本配置
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="80;86"

# 可选参数:
# - DSPMV_REQUIRE_CUDA=OFF     # 禁用 CUDA 要求，仅配置模式
# - CMAKE_INSTALL_PREFIX=/path # 自定义安装路径
# - CMAKE_BUILD_TYPE=Debug     # Debug 模式

cmake --build . -j$(nproc)
```

### 5. 安装到系统

```bash
# 从 build 目录
sudo cmake --install .

# 或使用预设 (如果有定义)
cmake --build --preset release --target install
```

默认安装路径:
- 头文件: `/usr/local/include/spmv/`
- 库文件: `/usr/local/lib/libspmv.a`

---

## 验证安装

### 运行测试套件

```bash
# 使用 CTest
ctest --preset default

# 或直接运行测试程序
./build-release/spmv_tests
```

### 运行基准测试

```bash
# 构建并运行基准测试
./build-release/spmv_benchmark
```

预期输出:
```
========================================
GPU SpMV Benchmark
========================================
GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Memory Bandwidth: 760.3 GB/s
...
```

### 验证库文件

```bash
# 检查静态库
ls -la /usr/local/lib/libspmv.a

# 检查头文件
ls /usr/local/include/spmv/
# 应显示: bandwidth.h benchmark.h common.h csr_matrix.h ...
```

---

## 常见问题

### Q: CMake 报错 "No CUDA toolset found"

**解决方案:**
```bash
# 确保 CUDA 在 PATH 中
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 或在 CMake 中指定 CUDA 路径
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda
```

### Q: 编译错误 "undefined reference to cuda..."

**解决方案:**
确保 CUDA 库在链接路径中:
```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Q: "GPU 架构不匹配" 警告

**解决方案:**
指定正确的 GPU 架构:
```bash
# 查询您的 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 在 CMake 中指定
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # 对于 8.6
```

### Q: 测试运行时 CUDA 错误

**可能原因:**
- 驱动版本与 CUDA 版本不兼容
- 没有可用的 GPU (在无 GPU 环境中运行)

**解决方案:**
```bash
# 检查驱动版本
nvidia-smi

# 在无 GPU 环境中使用 CPU-only 配置
cmake --preset no-cuda
```

---

## 卸载

```bash
# 如果通过 CMake 安装
sudo xargs rm < build/install_manifest.txt

# 手动删除
sudo rm -rf /usr/local/include/spmv/
sudo rm -f /usr/local/lib/libspmv.a
```

---

## 下一步

- [**📚 API 参考**](api) - 学习如何使用库接口
- [**📝 示例代码**](examples) - 查看完整示例程序
- [**🚀 性能优化**](performance) - 了解性能调优技巧

---

<div align="center">

**[← 返回首页](index)** · **[ API 参考 →](api)**

</div>
