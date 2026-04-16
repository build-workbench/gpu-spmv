---
layout: default
title: Installation
parent: Documentation
nav_order: 1
---

# 📦 Installation
{: .no_toc }

Installation and configuration guide.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Requirements

### Required Components

| Component | Minimum | Recommended |
|:----------|:-------:|:-----------:|
| **CUDA Toolkit** | 11.0 | 12.0+ |
| **CMake** | 3.18 | 3.25+ |
| **C++ Compiler** | C++17 | C++17 |
| **NVIDIA GPU** | CC 7.0 | CC 8.6+ |

---

## Ubuntu / Debian

```bash
# 1. Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# 2. Install build tools
sudo apt-get install -y cmake build-essential git

# 3. Verify
nvcc --version && nvidia-smi
```

---

## Build

### Using CMake Presets

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset release
cmake --build --preset release
```

### Install System-wide

```bash
sudo cmake --install build-release
```

---

## Verification

```bash
# Run tests
ctest --preset default

# Run benchmark
./build-release/spmv_benchmark
```
