---
layout: default
title: GPU SpMV Documentation
lang: en
---

<p align="right">
  <a href="index.html">🇨🇳 简体中文</a>
</p>

# GPU SpMV Documentation Center

**High-Performance CUDA Sparse Matrix-Vector Multiplication Library**

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue.svg)](https://github.com/LessUp/gpu-spmv/releases)

---

## Quick Navigation

| Document | Description | Target Audience |
|:--------:|:------------|:----------------|
| [**📦 Installation**](installation.en) | System requirements, build instructions, dependencies | New Users |
| [**📚 API Reference**](api.en) | Complete interface documentation, data structures, error handling | Developers |
| [**📝 Examples**](examples.en) | Basic usage, advanced features, complete applications | Learners |
| [**🚀 Performance**](performance.en) | Optimization strategies, benchmarking, best practices | Performance Engineers |
| [**🏗️ Architecture**](architecture.en) | System architecture, core algorithms, design decisions | Architects |
| [**📋 Changelog**](changelog.en) | Version history, migration guides, roadmap | Maintainers |

---

## Project Overview

GPU SpMV is a **Sparse Matrix-Vector Multiplication** library optimized for NVIDIA GPUs. It supports multiple sparse matrix storage formats and achieves excellent memory bandwidth utilization across various matrix distributions through intelligent kernel selection strategies.

### Key Features

<div class="feature-grid">

**🎯 Multi-Format Support**
- CSR (Compressed Sparse Row) - General purpose format
- ELL (ELLPACK) - GPU-optimized format

**⚡ Intelligent Kernel Selection**
- Automatic optimal kernel selection based on matrix statistics
- Four strategies: Scalar, Vector, Merge Path, and ELL

**📊 High-Performance Implementation**
- Warp-level parallel reduction
- Texture cache optimization
- Coalesced memory access patterns

**🔧 Production Quality**
- RAII resource management
- Semantic error codes
- Comprehensive test coverage

</div>

---

## 30-Second Quick Start

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

int main() {
    // 1. Create CSR matrix from dense data
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    // 2. Prepare input/output vectors
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);

    // 3. Auto-select optimal config and execute
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

    // 4. Retrieve results
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    // Result: y = [3, 7, 5]

    csr_destroy(csr);
    return 0;
}
```

---

## Performance Metrics

Typical performance on NVIDIA RTX 3080:

| Matrix Size | Non-zeros | Recommended Kernel | Bandwidth Utilization |
|:-----------:|:---------:|:-------------------|:---------------------:|
| 10K × 10K | 500K | Vector CSR | **~70%** |
| 100K × 100K | 5M | Merge Path | **~65%** |
| 1M × 1M | 50M | Merge Path | **~60%** |

> 💡 Actual performance depends on matrix structure and GPU model. Use `spmv_auto_config()` to automatically select the optimal strategy.

---

## Documentation Conventions

### Notation Guide

| Symbol | Meaning |
|:------:|:--------|
| `code` | Code snippets, function names, variable names |
| **bold** | Important concepts, emphasized content |
| [link]() | Clickable documentation links |
| ⚠️ Warning | Important considerations |
| 💡 Tip | Useful hints and suggestions |
| 📝 Note | Supplementary information |

### Version Information

This documentation corresponds to GPU SpMV **v1.0.0**. To view documentation for other versions, please visit [GitHub Releases](https://github.com/LessUp/gpu-spmv/releases).

---

## Getting Help

- **Issue Reporting**: [GitHub Issues](https://github.com/LessUp/gpu-spmv/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/LessUp/gpu-spmv/discussions)
- **Source Repository**: [GitHub Repository](https://github.com/LessUp/gpu-spmv)

---

## License

This project is licensed under the [MIT License](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE).

---

<div align="center">

**[Read Docs 📚](installation.en)** · **[View Examples 📝](examples.en)** · **[GitHub Repo 🐙](https://github.com/LessUp/gpu-spmv)**

</div>
