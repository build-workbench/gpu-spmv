---
layout: default
title: 安装指南
parent: 中文文档
nav_order: 1
lang: zh
---

<p align="right">
  <a href="installation.en">🇺🇸 English</a>
</p>

# 📦 安装指南
{: .no_toc }

本指南将帮助您在系统上安装和配置 GPU SpMV 库。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 系统要求

### 必需组件

| 组件 | 最低版本 | 推荐版本 |
|:-----|:--------:|:--------:|
| **CUDA Toolkit** | 11.0 | 12.0+ |
| **CMake** | 3.18 | 3.25+ |
| **C++ 编译器** | C++17 | C++17 |
| **NVIDIA GPU** | CC 7.0 | CC 8.6+ |

### 支持的 GPU 架构

| 架构 | 计算能力 | 代表型号 |
|:-----|:--------:|:---------|
| Volta | 7.0, 7.5 | V100, Titan V |
| Turing | 7.5 | RTX 20 系列, T4 |
| Ampere | 8.0, 8.6 | RTX 30 系列, A100 |
| Ada Lovelace | 8.9 | RTX 40 系列 |
| Hopper | 9.0 | H100 |

---

## 依赖安装

### Ubuntu / Debian

```bash
# 1. 安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# 2. 安装构建工具
sudo apt-get install -y cmake build-essential git

# 3. 验证安装
nvcc --version
nvidia-smi
```

### CentOS / RHEL / Rocky Linux

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git
```

### Windows

1. 安装 Visual Studio 2019/2022（C++ 桌面开发）
2. 安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. 安装 [CMake](https://cmake.org/download/)

---

## 构建步骤

### 使用 CMake Presets（推荐）

```bash
# 克隆仓库
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# Release 构建（生产环境）
cmake --preset release
cmake --build --preset release

# Debug 构建（开发调试）
cmake --preset default
cmake --build --preset default
```

### 自定义构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="80;86"
cmake --build . -j$(nproc)
```

### 安装到系统

```bash
sudo cmake --install .
```

---

## 验证安装

### 运行测试

```bash
ctest --preset default
# 或
./build-release/spmv_tests
```

### 运行基准测试

```bash
./build-release/spmv_benchmark
```

---

## 卸载

```bash
sudo xargs rm < build/install_manifest.txt
```
