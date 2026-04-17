---
layout: default
title: Home
nav_order: 2
has_children: false
lang: en
---

# GPU SpMV
{: .fs-9 .fw-700 .text-center }

High-Performance CUDA Sparse Matrix-Vector Multiplication Library
{: .fs-6 .fw-300 .text-center .text-grey-dk-500 }

<div class="text-center" style="margin: 2rem 0;">
  <a href="installation.en" class="btn btn-primary fs-5 mb-4 mb-md-0 mr-2">Quick Start →</a>
  <a href="api.en" class="btn btn-green fs-5 mb-4 mb-md-0 mr-2">API Docs</a>
  <a href="https://github.com/LessUp/gpu-spmv" class="btn fs-5 mb-4 mb-md-0">GitHub</a>
</div>

---

## ✨ Key Features

<div class="grid grid-cols-3 gap-4" markdown="1">

### 🚀 Extreme Performance

Multi-kernel intelligent scheduling, optimized for NVIDIA GPUs, up to 70%+ bandwidth utilization

### 📊 Multi-Format Support

CSR general format + ELL high-performance format, adapted for different sparsity matrices

### 🎯 Production Quality

RAII resource management, semantic error codes, complete test coverage, enterprise-grade reliability

</div>

---

## 🔥 Why Choose GPU SpMV?

### Intelligent Kernel Selection

Automatically selects the optimal kernel based on matrix characteristics:

```cpp
// Automatically analyze matrix features, select best kernel
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

| Matrix Characteristics | Recommended Kernel | Performance |
|:----------------------|:------------------|:------------|
| Very Sparse (avg_nnz < 4) | Scalar CSR | ★★★☆☆ |
| Moderate Sparsity (skewness < 10) | Vector CSR | ★★★★☆ |
| Highly Skewed (skewness ≥ 10) | Merge Path | ★★★★★ |
| ELL Format | ELL Kernel | ★★★★★ |

### Simple API Design

```cpp
#include <spmv/spmv.h>

// 1. Create CSR matrix
CSRMatrix* csr = csr_create(num_rows, num_cols, nnz);
csr_from_dense(csr, host_data, num_rows, num_cols);

// 2. Transfer to GPU
csr_to_gpu(csr);

// 3. Execute SpMV
SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

// 4. Clean up
csr_destroy(csr);
```

### Enterprise-Grade Features

<div class="grid grid-cols-2 gap-3" markdown="1">

#### 🎯 RAII Resource Management

Automatic lifecycle management, preventing memory leaks:
```cpp
CudaBuffer<float> buffer(1000);
// Automatically releases GPU memory
```

#### 🔍 Semantic Error Codes

Clear error tracing:
```cpp
if (result.error != SpMVError::SUCCESS) {
    printf("Error: %s\n", spmv_error_string(result.error));
}
```

#### 🖥️ Cross-Platform Support

Windows / Linux full platform support

#### 🔧 CMake Presets

One-click build, zero configuration:
```bash
cmake --preset release
cmake --build --preset release
```

#### ✅ Complete Test Coverage

Google Test + 100+ property test cases

#### 📈 Performance Benchmarks

Built-in benchmark tools, quantifiable performance metrics

</div>

---

## 📊 Performance

Test results on NVIDIA RTX 3090 (Ampere):

| Matrix Size | Non-Zero Elements | Kernel | Bandwidth Utilization |
|:-----------:|:-----------------:|:-------|:---------------------:|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

---

## 🚀 Quick Start

### System Requirements

- CUDA Toolkit 11.0+ / 12.0+
- CMake 3.18+
- C++17 Compiler
- NVIDIA GPU (CC 7.0+)

### Three-Step Installation

```bash
# 1. Clone repository
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# 2. Build
cmake --preset release
cmake --build --preset release

# 3. Test
ctest --preset default
```

---

## 📚 Documentation Navigation

<div class="grid grid-cols-3 gap-4" markdown="1">

### 📦 [Installation Guide](installation.en)

System requirements, dependency installation, build steps

### 🏗️ [Architecture Design](architecture.en)

System architecture, core algorithms, design decisions

### 📚 [API Reference](api.en)

Complete interface documentation, data structures, error handling

### 📝 [Code Examples](examples.en)

Basic usage, advanced features, complete applications

### 🚀 [Performance Optimization](performance.en)

Tuning strategies, benchmark testing, best practices

### 📋 [Changelog](changelog.en)

Version history, migration guide

</div>

---

## 🌟 Typical Use Cases

- **Graph Algorithms**: PageRank, shortest path
- **Scientific Computing**: Finite element analysis, computational fluid dynamics
- **Machine Learning**: Sparse neural networks, recommendation systems
- **Data Analytics**: Matrix decomposition, eigenvalue computation

---

## 🤝 Contributing

GPU SpMV is an open source project, contributions are welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>GPU SpMV is licensed under <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">MIT License</a></p>
  <p>Copyright &copy; 2024-2026 LessUp</p>
</div>
