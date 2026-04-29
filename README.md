<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B" alt="C++">
  <img src="https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue" alt="Platform">
</p>

<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>High-Performance CUDA Sparse Matrix-Vector Multiplication Library</strong>
</p>

<p align="center">
  <em>4 optimized kernels · 2 sparse formats · 70%+ bandwidth utilization · Production-ready</em>
</p>

<p align="center">
  <a href="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://lessup.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/Docs-GitHub%20Pages-2EA44F?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/releases">
    <img src="https://img.shields.io/github/v/release/LessUp/gpu-spmv?color=blue" alt="Release">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="README.md"><b>English</b></a> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a>
  · <a href="#-features">Features</a>
  · <a href="#-performance">Performance</a>
  · <a href="#-documentation">Documentation</a>
  · <a href="#-contributing">Contributing</a>
</p>

---

## 🎯 What is GPU SpMV?

GPU SpMV is a **production-ready C++ library** that accelerates sparse matrix-vector multiplication on NVIDIA GPUs. It automatically selects the optimal kernel based on matrix characteristics, delivering up to **70%+ of theoretical memory bandwidth**.

**Perfect for**: Graph algorithms · Scientific computing · Machine learning · Data analytics

---

## ✨ Why Choose GPU SpMV?

### 🚀 Intelligent Kernel Selection

Four optimized kernels with automatic selection based on matrix features:

| Matrix Pattern | Kernel | Strategy | Performance |
|:--------------|:-------|:---------|:-----------:|
| Very sparse (avg_nnz < 4) | Scalar CSR | 1 thread/row | ★★★☆☆ |
| Uniform (skewness < 10) | Vector CSR | 1 warp/row | ★★★★☆ |
| Skewed (skewness ≥ 10) | Merge Path | Perfect balance | ★★★★★ |
| ELL format | ELL Kernel | Coalesced access | ★★★★★ |

### 📊 Multi-Format Support

- **CSR** (Compressed Sparse Row) - General-purpose sparse matrices
- **ELL** (ELLPACK) - Uniform row lengths with maximum performance

### 🎯 Production-Grade Quality

```cpp
// RAII resource management - automatic cleanup
CudaBuffer<float> d_x(1000);  // GPU memory auto-freed
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

// Semantic error handling - clear diagnostics
if (result.error != SpMVError::SUCCESS) {
    printf("Error: %s\n", spmv_error_string(result.error));
}
```

- ✅ **RAII Management** - `CudaBuffer`, `SpMVExecutionContext`
- 🔍 **Error Codes** - Semantic `SpMVError` enum
- 🖥️ **Cross-Platform** - Windows & Linux
- 🔧 **Modern Build** - CMake Presets, one-click build
- ✅ **Full Testing** - Google Test + 100+ property tests

---

## 🚀 Quick Start

### Prerequisites

| Component | Minimum | Recommended |
|:----------|:-------:|:-----------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |

### 3-Step Installation

```bash
# 1. Clone
git clone https://github.com/LessUp/gpu-spmv.git && cd gpu-spmv

# 2. Build
cmake --preset release && cmake --build --preset release

# 3. Test
ctest --preset default  # All tests should pass ✅
```

⏱️ **Build time**: ~2 minutes on modern machine

### 💻 30-Second Example

```cpp
#include <spmv/spmv.h>

int main() {
    // 1. Create 3×3 sparse matrix: [1 0 2; 0 3 4; 0 0 5]
    float data[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, data, 3, 3);
    csr_to_gpu(csr);

    // 2. Prepare vectors
    CudaBuffer<float> d_x(3), d_y(3);
    float h_x[] = {1, 1, 1};
    cudaMemcpy(d_x.data(), h_x, sizeof(h_x), cudaMemcpyHostToDevice);

    // 3. Execute (auto-selects optimal kernel)
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    // result.time_ms ≈ 0.05ms, result.error == SUCCESS

    // 4. Get result: y = [3, 7, 5]
    csr_destroy(csr);
}
```

📚 **More examples**: [Documentation Site](https://lessup.github.io/gpu-spmv/examples)

---

## 📊 Performance

Benchmark on **NVIDIA RTX 3090** (Ampere, 936 GB/s peak):

| Matrix Size | NNZ | Kernel | Time | Bandwidth | Utilization |
|:-----------:|:---:|:-------|:----:|:---------:|:-----------:|
| 10K × 10K | 500K | Vector CSR | 2.3ms | 68.5 GB/s | **70.2%** |
| 100K × 100K | 5M | Merge Path | 23.5ms | 69.8 GB/s | **71.5%** |
| 1M × 1M | 50M | Merge Path | 235ms | 69.1 GB/s | **70.8%** |

```bash
# Run your own benchmarks
./build-release/spmv_benchmark

# Output example:
# GPU: NVIDIA GeForce RTX 3090
# Matrix: 100000x100000, NNZ: 5000000
# Avg time: 23.5 ms | Bandwidth: 69.8 GB/s (71.5% of peak)
```

📈 **Full performance guide**: [Performance Optimization](https://lessup.github.io/gpu-spmv/performance)

---

## 🏗️ Architecture

```
gpu-spmv/
├── include/spmv/          # Public headers (10 files)
│   ├── spmv.h             # Main SpMV interface
│   ├── csr_matrix.h       # CSR format
│   ├── ell_matrix.h       # ELL format
│   ├── cuda_buffer.h      # RAII GPU memory
│   ├── benchmark.h        # Performance testing
│   └── pagerank.h         # PageRank algorithm
├── src/                   # Implementations (7 files)
├── tests/                 # Google Test suite (8 files)
├── benchmarks/            # Performance benchmarks
└── specs/                 # SDD specifications
```

🔧 **Spec-Driven Development**: All features defined in [`/specs/`](specs/) before implementation

---

## 📚 Documentation

Complete documentation is available at **[https://lessup.github.io/gpu-spmv/](https://lessup.github.io/gpu-spmv/)**:

| Document | Description |
|:---------|:------------|
| [📦 Installation Guide](https://lessup.github.io/gpu-spmv/installation) | System requirements, detailed installation |
| [📚 API Reference](https://lessup.github.io/gpu-spmv/api) | Complete API documentation, data structures |
| [📝 Examples](https://lessup.github.io/gpu-spmv/examples) | 7 complete code examples (basic → advanced) |
| [🚀 Performance Guide](https://lessup.github.io/gpu-spmv/performance) | Tuning strategies, benchmark data |
| [🏗️ Architecture](https://lessup.github.io/gpu-spmv/architecture) | System design, kernel selection |
| [📋 Changelog](https://lessup.github.io/gpu-spmv/changelog) | Version history, migration guide |

---

## 🧪 Testing

```bash
# Run all tests
ctest --preset default

# Or run directly
./build-release/spmv_tests

# Run specific tests
./build-release/spmv_tests --gtest_filter="CSR*"
./build-release/spmv_tests --gtest_filter="ELL*"
```

**Test Coverage**:
- ✅ CSR/ELL format conversion
- ✅ SpMV computation correctness (vs CPU reference)
- ✅ Dimension validation
- ✅ Kernel selection logic
- ✅ Bandwidth metrics
- ✅ PageRank invariants
- ✅ 100+ property-based tests with random matrices

---

## 💡 Real-World Application: PageRank

```cpp
#include <spmv/pagerank.h>

// Build adjacency matrix for graph
CSRMatrix* adj = build_graph_adjacency();
csr_to_gpu(adj);

// Run PageRank
PageRankConfig config = {.damping = 0.85f, .tolerance = 1e-6f};
PageRankResult result = pagerank(adj, &config);

// Get top-10 ranked nodes
auto top_10 = get_top_k(result, 10);
for (const auto& node : top_10) {
    printf("Node %d: %.6f\n", node.id, node.rank);
}

pagerank_free(&result);
csr_destroy(adj);
```

📊 **Use cases**: Social network analysis · Web search · Recommendation systems · Fraud detection

---

## 🤝 Contributing

We welcome contributions! GPU SpMV follows **Spec-Driven Development** - specs are the single source of truth.

### Quick Contributing Guide

1. 🍴 **Fork** the repository
2. 📖 **Read specs** in `/specs/` for the feature you want
3. 🌿 **Create branch** (`git checkout -b feature/your-feature`)
4. 📝 **Update specs first** (if modifying behavior)
5. 💻 **Implement code** following spec
6. ✅ **Run tests** (`ctest --preset default`)
7. 🚀 **Open PR** with spec changes

📋 **Full guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

### Development Setup

```bash
# Format code (required before commit)
find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  | xargs clang-format -i

# Build & test
cmake --preset default && cmake --build --preset default && ctest --preset default
```

---

## 📄 License

MIT License © 2024-2026 LessUp. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Algorithm based on [Merge-based Parallel SpMV](https://research.nvidia.com/publication/merge-based-parallel-sparse-matrix-vector-multiplication) by Merrill & Garland (NVIDIA)
- CUDA optimizations from NVIDIA official documentation
- Inspired by cuSPARSE and modern sparse library design patterns

---

<p align="center">
  <sub>Built with ❤️ by the GPU SpMV contributors</sub>
</p>

<p align="center">
  <a href="#-quick-start">⬆️ Back to Top</a>
</p>
