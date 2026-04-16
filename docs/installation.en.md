---
layout: default
title: Installation Guide
lang: en
---

<p align="right">
  <a href="installation.html">🇨🇳 简体中文</a>
</p>

# 📦 Installation Guide

This guide will help you install and configure the GPU SpMV library on your system.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Dependency Installation](#dependency-installation)
- [Build Instructions](#build-instructions)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Uninstallation](#uninstallation)

---

## System Requirements

### Required Components

| Component | Minimum | Recommended | Notes |
|:----------|:-------:|:-----------:|:------|
| **CUDA Toolkit** | 11.0 | 12.0+ | Required for NVIDIA GPU computing |
| **CMake** | 3.18 | 3.25+ | Build system |
| **C++ Compiler** | C++17 | C++17 | GCC 7+ / Clang 5+ / MSVC 2017+ |
| **NVIDIA GPU** | CC 7.0 | CC 8.6+ | Volta architecture or newer |

### Supported GPU Architectures

| Architecture | Compute Capability | Representative GPUs |
|:-------------|:------------------:|:--------------------|
| Volta | 7.0, 7.5 | V100, Titan V |
| Turing | 7.5 | RTX 20 Series, T4 |
| Ampere | 8.0, 8.6 | RTX 30 Series, A100, A6000 |
| Ada Lovelace | 8.9 | RTX 40 Series |
| Hopper | 9.0 | H100 |

### Optional Components

| Component | Purpose |
|:----------|:--------|
| **NVIDIA Driver** | 450.80.02+ (CUDA 11) / 525.85.12+ (CUDA 12) |
| **Git** | Clone source repository |
| **Google Test** | Run unit tests (auto-downloaded) |

---

## Dependency Installation

### Ubuntu / Debian

```bash
# 1. Install CUDA Toolkit (if not installed)
# Method A: Via NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# Method B: Use existing CUDA version
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# 2. Install build tools
sudo apt-get install -y cmake build-essential git

# 3. Verify CUDA installation
nvcc --version
nvidia-smi
```

### CentOS / RHEL / Rocky Linux

```bash
# 1. Install development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git

# 2. Install CUDA (via runfile or NVIDIA repository)
# Reference: https://developer.nvidia.com/cuda-downloads

# 3. Verify installation
nvcc --version
```

### Windows

1. **Install Visual Studio 2019/2022**
   - Select "Desktop development with C++" workload
   - Ensure MSVC v142 or later is installed

2. **Install CUDA Toolkit**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select the corresponding Windows version

3. **Install CMake**
   - Download installer from [CMake website](https://cmake.org/download/)
   - Add to system PATH

4. **Verify Installation**
   ```cmd
   nvcc --version
   cmake --version
   ```

### macOS

> ⚠️ **Note**: macOS has discontinued CUDA support for NVIDIA GPUs. Use CPU-only mode for development, or switch to Linux/Windows environment.

---

## Build Instructions

### 1. Clone Repository

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
```

### 2. Configure Build (CMake Presets)

This project provides pre-configured CMake Presets (recommended):

```bash
# Release build (recommended for production)
cmake --preset release

# Debug build (for development)
cmake --preset default

# CPU-only mode (no GPU environment)
cmake --preset no-cuda
```

### 3. Execute Build

```bash
# Build using corresponding preset
cmake --build --preset release
```

### 4. Custom Build Options

For custom build parameters:

```bash
mkdir build && cd build

# Basic configuration
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES="80;86"

# Optional parameters:
# - DSPMV_REQUIRE_CUDA=OFF     # Disable CUDA requirement, configure-only mode
# - CMAKE_INSTALL_PREFIX=/path # Custom installation path
# - CMAKE_BUILD_TYPE=Debug     # Debug mode

cmake --build . -j$(nproc)
```

### 5. Install to System

```bash
# From build directory
sudo cmake --install .

# Or using preset (if defined)
cmake --build --preset release --target install
```

Default installation paths:
- Headers: `/usr/local/include/spmv/`
- Library: `/usr/local/lib/libspmv.a`

---

## Verification

### Run Test Suite

```bash
# Using CTest
ctest --preset default

# Or run test program directly
./build-release/spmv_tests
```

### Run Benchmark

```bash
# Build and run benchmark
./build-release/spmv_benchmark
```

Expected output:
```
========================================
GPU SpMV Benchmark
========================================
GPU: NVIDIA GeForce RTX 3080
Compute Capability: 8.6
Memory Bandwidth: 760.3 GB/s
...
```

### Verify Library Files

```bash
# Check static library
ls -la /usr/local/lib/libspmv.a

# Check headers
ls /usr/local/include/spmv/
# Should show: bandwidth.h benchmark.h common.h csr_matrix.h ...
```

---

## Troubleshooting

### Q: CMake error "No CUDA toolset found"

**Solution:**
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or specify CUDA path in CMake
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda
```

### Q: Link error "undefined reference to cuda..."

**Solution:**
Ensure CUDA libraries are in link path:
```bash
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

### Q: "GPU architecture mismatch" warning

**Solution:**
Specify correct GPU architecture:
```bash
# Query your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Specify in CMake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # For 8.6
```

### Q: CUDA errors during test execution

**Possible causes:**
- Driver version incompatible with CUDA version
- No available GPU (running in GPU-less environment)

**Solution:**
```bash
# Check driver version
nvidia-smi

# Use CPU-only configuration in GPU-less environment
cmake --preset no-cuda
```

---

## Uninstallation

```bash
# If installed via CMake
sudo xargs rm < build/install_manifest.txt

# Manual removal
sudo rm -rf /usr/local/include/spmv/
sudo rm -f /usr/local/lib/libspmv.a
```

---

## Next Steps

- [**📚 API Reference**](api.en) - Learn how to use library interfaces
- [**📝 Examples**](examples.en) - View complete example programs
- [**🚀 Performance**](performance.en) - Learn performance optimization tips

---

<div align="center">

**[← Back to Home](index.en)** · **[ API Reference →](api.en)**

</div>
