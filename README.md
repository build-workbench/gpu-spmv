<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>High-Performance CUDA Sparse Matrix-Vector Multiplication Library</strong>
</p>

<p align="center">
  <a href="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://lessup.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/releases">
    <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  </a>
  <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
</p>

<p align="center">
  <b>English</b> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  Supported formats: <b>CSR</b>, <b>ELL</b> | Kernels: <b>Scalar</b>, <b>Vector</b>, <b>Merge Path</b> | Auto-selection | Production-ready
</p>

---

## ✨ Features

### Multiple Sparse Matrix Formats

| Format | Description | Best For | GPU Efficiency |
|:-------|:------------|:---------|:--------------:|
| **CSR** | Compressed Sparse Row | General-purpose sparse matrices | ★★★☆☆ |
| **ELL** | ELLPACK | Uniform row lengths | ★★★★★ |

### Four Optimized CUDA Kernels

```
┌─────────────────────────────────────────────────────────────────┐
│              Automatic Kernel Selection Strategy                 │
├─────────────────────────────────────────────────────────────────┤
│  avg_nnz_per_row < 4     ──→  Scalar CSR                        │
│  (Very sparse rows)           One thread per row                │
│                                                                 │
│  skewness < 10           ──→  Vector CSR                        │
│  (Uniform distribution)       One warp (32 threads) per row     │
│                                                                 │
│  skewness >= 10          ──→  Merge Path                        │
│  (Highly irregular)           Perfect load balancing            │
│                                                                 │
│  ELL format              ──→  ELL Kernel                        │
│  (Uniform rows)               Column-major coalesced access     │
└─────────────────────────────────────────────────────────────────┘
```

### Production-Ready Engineering

- 🎯 **RAII Resource Management** — `CudaBuffer`, `CudaTimer`, `SpMVExecutionContext` for automatic GPU resource handling
- 🔍 **Semantic Error Codes** — `SpMVError` enum + `CUDA_CHECK_*` macros for clear error tracking
- 🖥️ **Cross-Platform Support** — Automatic Windows/Linux test path adaptation
- 🔧 **Modern Build System** — CMake Presets for one-click Debug/Release builds
- ✅ **CI/CD Ready** — GitHub Actions with format checking and build verification

---

## 🚀 Quick Start

### Requirements

| Component | Minimum | Recommended |
|:----------|:-------:|:-----------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Standard | C++17 | C++17 |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |

### Installation

```bash
# Clone repository
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# Build (Release recommended)
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset default
```

### 30-Second Example

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

int main() {
    // 1. Create sparse matrix from dense data
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    // 2. Prepare vectors
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);

    // 3. Execute SpMV (auto-selects optimal kernel)
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

    // 4. Get results
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    // Output: y = [3, 7, 5]

    csr_destroy(csr);
    return 0;
}
```

---

## 📊 Performance

Benchmark results on NVIDIA RTX 3080:

| Matrix Size | Non-zeros | Kernel | Bandwidth Utilization |
|:-----------:|:---------:|:-------|:---------------------:|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

```bash
# Run benchmark
./build-release/spmv_benchmark

# Example output:
# GPU: NVIDIA GeForce RTX 3080
# Matrix: 100000x100000, NNZ: 5000000
# Avg time: 0.312 ms
# Bandwidth: 495.2 GB/s (65.1% of peak)
```

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| [📖 **Documentation Site**](https://lessup.github.io/gpu-spmv/) | Full documentation with examples |
| [📦 **Installation**](https://lessup.github.io/gpu-spmv/installation.en) | Detailed installation guide |
| [📚 **API Reference**](https://lessup.github.io/gpu-spmv/api.en) | Complete API documentation |
| [🚀 **Performance Guide**](https://lessup.github.io/gpu-spmv/performance.en) | Optimization strategies |
| [📝 **Examples**](https://lessup.github.io/gpu-spmv/examples.en) | Code examples collection |
| [📋 **Changelog**](https://lessup.github.io/gpu-spmv/changelog.en) | Version history & migration guide |

---

## 🏗️ Architecture

```
gpu-spmv/
├── include/spmv/          # Public headers
│   ├── common.h           # Error codes, CUDA macros
│   ├── cuda_buffer.h      # RAII GPU memory management
│   ├── csr_matrix.h       # CSR sparse matrix format
│   ├── ell_matrix.h       # ELL sparse matrix format
│   ├── spmv.h             # SpMV interface & kernel selection
│   ├── bandwidth.h        # Bandwidth metrics
│   ├── benchmark.h        # Benchmarking framework
│   └── pagerank.h         # PageRank algorithm
├── src/                   # Source implementations
├── tests/                 # Property-based tests + unit tests
├── benchmarks/            # Performance benchmarks
└── docs/                  # Documentation
```

---

## 🧪 Testing

```bash
# Run all tests
./build-release/spmv_tests

# Run specific test
./build-release/spmv_tests --gtest_filter="CSR*"

# Property tests with random matrices
./build-release/spmv_tests --gtest_repeat=10
```

Test coverage includes:
- ✅ CSR/ELL format conversion correctness
- ✅ SpMV computation verification (vs. CPU reference)
- ✅ Dimension validation
- ✅ Kernel selector validity
- ✅ Bandwidth metric validation
- ✅ PageRank invariant checking

---

## 💡 Application Example: PageRank

```cpp
#include "spmv/pagerank.h"
#include "spmv/csr_matrix.h"

// Create column-normalized adjacency matrix
CSRMatrix* adj = create_normalized_adjacency();
csr_to_gpu(adj);

// Configure PageRank
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

// Run PageRank
PageRankResult result = pagerank(adj, &config);

// Get top-10 nodes
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Format code
find src tests include -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i

# Build and test
cmake --preset default
cmake --build --preset default
ctest --preset default
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- Inspired by [Merge-based Parallel Sparse Matrix-Vector Multiplication](https://research.nvidia.com/sites/default/files/pubs/2014-09_Merge-based-Parallel-Sparse/merge-based-spmv.pdf) by Merrill & Garland
- CUDA optimization techniques from NVIDIA documentation

---

<div align="center">

**[📖 Documentation](https://lessup.github.io/gpu-spmv/)** · **[🚀 Getting Started](https://lessup.github.io/gpu-spmv/installation.en)** · **[💻 GitHub](https://github.com/LessUp/gpu-spmv)**

</div>
