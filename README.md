# GPU SpMV (Sparse Matrix-Vector Multiplication)

[![CI](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/gpu-spmv/actions/workflows/ci.yml)

English | [简体中文](README.zh-CN.md)

High-performance CUDA sparse matrix-vector multiplication library supporting CSR and ELL formats with multiple load-balancing optimization strategies.

## Features

- **Multiple Sparse Formats** — CSR and ELL (ELLPACK)
- **Optimized CUDA Kernels** — Scalar CSR, Vector CSR (warp-per-row), Merge Path, ELL
- **Auto Kernel Selection** — Based on matrix characteristics
- **Performance Metrics** — Bandwidth utilization, GFLOPS, benchmark framework
- **Engineering Quality** — RAII (`CudaBuffer`, `CudaTimer`, `ScopedTexture`), semantic error codes, CMake Presets, CI

## Quick Start

```bash
# Using CMake Presets
cmake --preset default
cmake --build --preset default

# Run tests from the default preset build tree
ctest --preset default

# Or build release artifacts
cmake --preset release
cmake --build --preset release

# Run benchmarks (requires a CUDA-capable GPU)
./build/spmv_benchmark
# release preset output:
./build-release/spmv_benchmark
```

If you only want CMake configuration to succeed without generating CUDA build targets, use:

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
```

That mode is configure-only and does not produce `spmv`, `spmv_tests`, or `spmv_benchmark`.

> `spmv_benchmark` now exits early with a clear error when no usable CUDA device is available.


## Requirements

- CUDA Toolkit 11.0+, CMake 3.18+, C++17, GPU CC 7.0+

## Usage

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, dense.data(), 3, 3);
csr_to_gpu(csr);

SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
```

## Kernel Selection Strategy

- **Short rows (avg_nnz < 4)**: Scalar CSR
- **Uniform distribution (skewness < 10)**: Vector CSR
- **Highly skewed (skewness >= 10)**: Merge Path

## Project Structure

```
├── include/spmv/       # Headers (common, cuda_buffer, csr, ell, spmv, benchmark, pagerank)
├── src/                # Source files
├── tests/              # Property tests + unit tests
├── benchmarks/         # Benchmark program
├── CMakePresets.json   # Build presets
└── .github/workflows/  # CI
```

## License

MIT License
