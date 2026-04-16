# GPU SpMV (Sparse Matrix-Vector Multiplication)

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/gpu-spmv/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/LessUp/gpu-spmv/blob/main/LICENSE)

English | [简体中文](README.zh-CN.md)

**High-performance CUDA sparse matrix-vector multiplication library**

Supporting CSR and ELL formats with multiple load-balancing optimization strategies and automatic kernel selection.

---

## Core Features

### Multiple Sparse Matrix Formats

| Format | Description | Best For |
|--------|-------------|----------|
| **CSR** | Compressed Sparse Row | General-purpose, most sparse matrices |
| **ELL** | ELLPACK | Uniform row lengths, optimal GPU memory access |

### Four Optimized CUDA Kernels

```
┌─────────────────────────────────────────────────────────────┐
│               Automatic Kernel Selection Strategy            │
├─────────────────────────────────────────────────────────────┤
│  avg_nnz_per_row < 4     ──→  Scalar CSR                    │
│  (few elements per row)       One thread per row            │
│                                                             │
│  skewness < 10           ──→  Vector CSR                    │
│  (uniform row lengths)        One warp (32 threads) per row │
│                                                             │
│  skewness >= 10          ──→  Merge Path                    │
│  (highly irregular)           Perfect load balancing        │
│                                                             │
│  Row length ≈ max_nnz    ──→  ELL Kernel                    │
│  (ELL format)                 Column-major coalesced access │
└─────────────────────────────────────────────────────────────┘
```

### Engineering Quality

- **RAII Resource Management** — `CudaBuffer`, `CudaTimer`, `ScopedTexture` for automatic GPU resource handling
- **Semantic Error Codes** — `SpMVError` enum + `CUDA_CHECK_*` macros for clear error tracking
- **Cross-Platform Support** — Automatic Windows/Linux test path adaptation
- **Modern Build System** — CMake Presets for one-click Debug/Release builds
- **Continuous Integration** — GitHub Actions with automatic format checking and build verification

---

## Quick Start

### Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- NVIDIA GPU (Compute Capability 7.0+)

### Build & Install

```bash
# Clone repository
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

# Release build (recommended)
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
    // 1. Create sparse matrix
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    // 2. Prepare input vector
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
    printf("Result: [%.0f, %.0f, %.0f]\n", y[0], y[1], y[2]);
    
    csr_destroy(csr);
    return 0;
}
```

---

## Application Examples

### PageRank Graph Algorithm

```cpp
#include "spmv/pagerank.h"

// Create column-normalized adjacency matrix
CSRMatrix* adj = create_adjacency_matrix();
csr_to_gpu(adj);

// Configure PageRank parameters
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;
config.max_iterations = 100;

// Run PageRank
PageRankResult result = pagerank(adj, &config);

// Get Top-10 nodes
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

### Performance Benchmarking

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig config;
config.num_warmup_runs = 5;
config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x.data(), nullptr, &config);

printf("Average time: %.3f ms\n", result.avg_time_ms);
printf("GFLOPS: %.2f\n", result.gflops);
printf("Bandwidth: %.1f GB/s (%.1f%% efficiency)\n", 
       result.bandwidth_gb_s,
       result.bandwidth_gb_s / get_gpu_peak_bandwidth() * 100);
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [**API Reference**](https://lessup.github.io/gpu-spmv/api) | Complete API documentation: data structures, interfaces, error codes |
| [**Performance Guide**](https://lessup.github.io/gpu-spmv/performance) | Kernel selection strategies, bandwidth optimization, benchmarking |
| [**Code Examples**](https://lessup.github.io/gpu-spmv/examples) | Complete examples: basic usage, format conversion, PageRank, solvers |
| [**Changelog**](CHANGELOG.md) | Version history, change records, migration guide |

---

## Performance Overview

Performance on typical sparse matrices:

| Matrix Size | Non-zeros | Kernel | Bandwidth Utilization |
|-------------|-----------|--------|----------------------|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

> Actual performance depends on matrix structure and GPU model. Use `spmv_auto_config()` to automatically select the optimal strategy.

---

## Project Structure

```
gpu-spmv/
├── include/spmv/          # Public headers
│   ├── common.h           # Error codes, CUDA_CHECK_* macros
│   ├── cuda_buffer.h      # RAII GPU memory management
│   ├── csr_matrix.h       # CSR sparse matrix
│   ├── ell_matrix.h       # ELL sparse matrix
│   ├── spmv.h             # SpMV interface, kernel selection
│   ├── bandwidth.h        # Bandwidth metrics
│   ├── benchmark.h        # Benchmarking framework
│   └── pagerank.h         # PageRank algorithm
├── src/                   # Implementation
├── tests/                 # Property tests + unit tests
├── benchmarks/            # Performance benchmarks
└── docs/                  # Online documentation
```

---

## Testing

The project includes a comprehensive property-based test suite verifying:

- CSR/ELL format conversion correctness
- SpMV computation correctness (vs. CPU reference)
- Dimension validation
- Kernel selector validity
- Bandwidth metric validity
- PageRank invariants

Each property test runs 100 iterations with randomly generated matrices.

---

## License

[MIT License](LICENSE)

---

## Contributing

Issues and Pull Requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
