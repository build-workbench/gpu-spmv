---
layout: default
title: Changelog
lang: en
---

<p align="right">
  <a href="changelog.html">🇨🇳 简体中文</a>
</p>

# 📋 Changelog

All notable changes to the GPU SpMV project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-04-16

### 🎉 First Stable Release

#### ✨ Added

**Core Features**
- Full CSR (Compressed Sparse Row) sparse matrix format support
- Full ELL (ELLPACK) sparse matrix format support
- Four CUDA kernels: Scalar CSR, Vector CSR, Merge Path, ELL
- Automatic kernel selection based on matrix statistics
- Texture cache support with execution context reuse

**Performance & Testing**
- Bandwidth metrics calculation
- Benchmarking framework
- GPU vs CPU performance comparison

**Applications**
- PageRank graph algorithm implementation

**Engineering Quality**
- RAII resource management (`CudaBuffer`, `CudaTimer`, `SpMVExecutionContext`)
- Semantic error code system
- Comprehensive Google Test suite
- CMake Presets build configuration
- GitHub Actions CI/CD

#### 🔒 Security
- Integer overflow protection
- Memory safety checks

#### 🚀 Performance
- ELL Column-major coalesced memory access
- Warp-level shuffle reduction
- Load balancing optimization (Merge Path)

---

## [0.1.0] - 2025-03-01

### 🚀 Initial Release

- Basic project structure
- Basic CSR matrix implementation
- Simple SpMV GPU kernel
- CMake build configuration

---

## Version History

| Version | Date | Status | Highlights |
|:-------:|:----:|:------:|:-----------|
| 1.0.0 | 2025-04-16 | Stable | First stable release, complete feature set |
| 0.1.0 | 2025-03-01 | Archived | Initial prototype |

---

## Migration Guide

### Upgrading to 1.0.0

No breaking changes from pre-release versions.

#### Recommended Updates

1. **Use Named Constants**
   ```cpp
   // Old
   config.block_size = 256;
   
   // New
   config.block_size = spmv::DEFAULT_BLOCK_SIZE;
   ```

2. **Use Execution Context for Reuse**
   ```cpp
   // Old
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols);
   }
   
   // New
   SpMVExecutionContext context;
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols, &context);
   }
   ```

---

## Future Roadmap

### Planned [1.1.0]
- [ ] COO format support
- [ ] Hybrid CSR/ELL format
- [ ] Multi-GPU support
- [ ] Batched SpMV operations

### Under Consideration
- [ ] BFloat16 precision support
- [ ] Automatic format selection tuning
- [ ] Python bindings

---

<div align="center">

**[← Performance](performance.en)** · **[ 🏠 Home →](index.en)**

</div>
