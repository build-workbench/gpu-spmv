# GPU SpMV (Sparse Matrix-Vector Multiplication)

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/gpu-spmv/)

English | [简体中文](README.zh-CN.md)

High-performance CUDA sparse matrix-vector multiplication library supporting CSR and ELL formats with multiple load-balancing optimization strategies.

## Features

- **Multiple Sparse Formats**
  - CSR (Compressed Sparse Row)
  - ELL (ELLPACK)

- **Optimized CUDA Kernels**
  - Scalar CSR: one thread per row
  - Vector CSR: one warp (32 threads) per row
  - Merge Path: balanced work distribution for highly irregular matrices
  - ELL Kernel: column-major access for uniform row lengths

- **Auto Kernel Selection** — automatically selects the optimal kernel based on matrix characteristics

- **Performance Metrics**
  - Bandwidth utilization analysis (peak bandwidth cached via `std::call_once`)
  - GFLOPS computation
  - Full benchmarking framework (CPU timing via `std::chrono`)
  - Optional texture cache for input vector reads

- **Engineering Quality**
  - RAII resource management (`CudaBuffer`, `CudaTimer`, `ScopedTexture`)
  - Semantic error codes (`CUDA_CHECK_MALLOC` / `CUDA_CHECK_MEMCPY`)
  - Cross-platform test paths (Windows / Linux)
  - CMake Presets
  - `.clang-format` + `.editorconfig`
  - GitHub Actions CI

- **Applications**
  - PageRank graph algorithm

## Build

### Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- NVIDIA GPU (Compute Capability 7.0+)

### Using CMake Presets (recommended)

```bash
# Debug build
cmake --preset default
cmake --build --preset default

# Release build
cmake --preset release
cmake --build --preset release

# Minimal build (sm_80 only)
cmake --preset minimal
cmake --build --preset minimal
```

### CPU-only configuration

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
```

This is configure-only and does not produce `spmv`, `spmv_tests`, or `spmv_benchmark`.

### Run tests

```bash
ctest --preset default
# or directly
./build/spmv_tests
```

### Run benchmarks

```bash
./build/spmv_benchmark
```

> `spmv_benchmark` exits early with a clear error when no CUDA device is available.

## Usage

### Basic SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

// Create from dense matrix
std::vector<float> dense = {
    1, 0, 2,
    0, 3, 4,
    0, 0, 5
};

CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, dense.data(), 3, 3);
csr_to_gpu(csr);

// Input vector
std::vector<float> x = {1, 1, 1};
CudaBuffer<float> d_x(3);
CudaBuffer<float> d_y(3);
d_x.copyFromHost(x.data(), 3);

// Execute SpMV
SpMVConfig config = spmv_auto_config(csr);
// config.use_texture = true;  // optional: enable texture cache for x
SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

// Get result
std::vector<float> y(3);
d_y.copyToHost(y.data(), 3);

csr_destroy(csr);
```

### PageRank

```cpp
#include "spmv/pagerank.h"

// Create adjacency matrix (column-normalized)
CSRMatrix* adj = /* ... */;
csr_to_gpu(adj);

// Run PageRank
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

PageRankResult result = pagerank(adj, &config);

// Get Top-10 nodes
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

### Benchmark

```cpp
#include "spmv/benchmark.h"
#include "spmv/common.h"

BenchmarkConfig bench_config;
bench_config.num_warmup_runs = 5;
bench_config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x.data(), &spmv_config, &bench_config);
if (result.error_code == static_cast<int>(SpMVError::SUCCESS)) {
    std::cout << "Avg time: " << result.avg_time_ms << " ms\n";
    std::cout << "GFLOPS: " << result.gflops << "\n";
    std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";

    // Export JSON
    std::string json = benchmark_to_json(result);
} else {
    std::cout << "Benchmark failed: "
              << spmv_error_string(static_cast<SpMVError>(result.error_code))
              << "\n";
}
```

## Performance

### Kernel Selection Strategy

| Condition | Kernel |
|---|---|
| Short rows (`avg_nnz < 4`) | Scalar CSR |
| Uniform distribution (`skewness < 10`) | Vector CSR |
| Highly skewed (`skewness >= 10`) | Merge Path |

### Bandwidth Optimizations

- Column-major storage (ELL format)
- Coalesced memory access
- Bandwidth utilization monitoring

## Testing

The project includes a comprehensive property-based test suite verifying:

- CSR/ELL format conversion correctness
- SpMV computation correctness (vs. CPU reference)
- Dimension validation
- Kernel selector validity
- Bandwidth metric validity
- PageRank invariants

Each property test runs 100 iterations with randomly generated matrices.

## Project Structure

```
.
├── include/spmv/       # Headers
│   ├── common.h        # Error codes, CUDA_CHECK_* macros
│   ├── cuda_buffer.h   # RAII GPU memory (memset/fill/bytes)
│   ├── csr_matrix.h    # CSR sparse matrix
│   ├── ell_matrix.h    # ELL sparse matrix (with nnz field)
│   ├── spmv.h          # SpMV interface & auto kernel selection
│   ├── bandwidth.h     # Bandwidth metrics
│   ├── benchmark.h     # Benchmarking framework
│   ├── pagerank.h      # PageRank algorithm
│   └── test_utils.h    # Test utilities (cross-platform paths)
├── src/                # Source files
│   ├── csr_matrix.cpp
│   ├── ell_matrix.cpp
│   ├── spmv_cpu.cpp
│   ├── spmv_kernels.cu  # RAII helpers: CudaTimer, ScopedTexture
│   ├── bandwidth.cpp    # Peak bandwidth cache (std::call_once)
│   ├── benchmark.cu     # CPU timing via std::chrono
│   └── pagerank.cu
├── tests/              # Property tests + unit tests
├── benchmarks/         # Benchmark program
├── .clang-format       # Code style
├── .editorconfig       # Editor config
├── CMakePresets.json   # Build presets
└── .github/workflows/  # CI
```

## Documentation

Project docs are published via GitHub Pages:

> **https://lessup.github.io/gpu-spmv/**

- [API Reference](https://lessup.github.io/gpu-spmv/api) — header interfaces, data structures, and function descriptions
- [Performance](https://lessup.github.io/gpu-spmv/performance) — kernel selection strategies, bandwidth optimizations, and benchmarks

## License

MIT License
