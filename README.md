<p align="center">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B" alt="C++">
  <img src="https://img.shields.io/badge/CMake-3.18%2B-064F8C?logo=cmake" alt="CMake">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue" alt="Platform">
</p>

<h1 align="center">GPU SpMV</h1>

<p align="center">
  <strong>Focused CUDA sparse matrix-vector multiplication library</strong>
</p>

<p align="center">
  <em>CSR + ELL formats · 4 kernels · explicit errors · minimal maintenance surface</em>
</p>

<p align="center">
  <a href="https://github.com/AICL-Lab/gpu-spmv/actions/workflows/ci.yml">
    <img src="https://github.com/AICL-Lab/gpu-spmv/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://aicl-lab.github.io/gpu-spmv/">
    <img src="https://img.shields.io/badge/Docs-GitHub%20Pages-2EA44F?logo=github" alt="Documentation">
  </a>
  <a href="https://github.com/AICL-Lab/gpu-spmv/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
</p>

<p align="center">
  <a href="README.md"><b>English</b></a> · <a href="README.zh-CN.md">简体中文</a>
</p>

## What it is

GPU SpMV is a C++17 / CUDA library for sparse matrix-vector multiplication on NVIDIA GPUs. The repository now concentrates on the core library only:

- **Storage**: CSR and ELL sparse formats
- **Execution**: Scalar CSR, Vector CSR, Merge Path, and ELL kernels
- **Engineering**: `CudaBuffer<T>` RAII, explicit `SpMVError`, CPU reference paths, focused tests

Non-core showcase modules and AI governance layers have been removed to keep the codebase smaller and easier to maintain.

## Quick start

```bash
git clone https://github.com/AICL-Lab/gpu-spmv.git
cd gpu-spmv

cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

CPU-only environments can use:

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda
ctest --test-dir build-no-cuda --output-on-failure
```

On Linux, GPU builds now have first-class presets that pin the system GCC/G++ host toolchain and
avoid Conda compiler leakage:

```bash
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

For release builds:

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## Minimal example

A complete, buildable version lives in [`examples/basic_spmv.cpp`](examples/basic_spmv.cpp)
(built by default; falls back to the CPU path in no-CUDA builds):

```cpp
#include <spmv/csr_matrix.h>
#include <spmv/cuda_buffer.h>
#include <spmv/spmv.h>

int main() {
    float dense[] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 4.0f,
        0.0f, 0.0f, 5.0f,
    };

    spmv::CSRMatrix* csr = spmv::csr_create(3, 3, 5);
    spmv::csr_from_dense(csr, dense, 3, 3);
    spmv::csr_to_gpu(csr);

    spmv::CudaBuffer<float> d_x(3);
    spmv::CudaBuffer<float> d_y(3);
    const float h_x[] = {1.0f, 1.0f, 1.0f};
    d_x.copyFromHost(h_x, 3);

    spmv::SpMVConfig config = spmv::spmv_auto_config(csr);
    spmv::SpMVResult result = spmv::spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);
    spmv::csr_destroy(csr);

    return result.error_code == 0 ? 0 : 1;
}
```

By default `spmv_csr` blocks until `d_y` is complete and reports timing
metrics. Set `config.enable_timing = false` to only enqueue the kernel on the
stream (no events, no sync) for pipelining; synchronize yourself before
reading `d_y`.

## Project layout

```text
gpu-spmv/
├── include/spmv/   # Public headers
├── src/            # Core library implementation
├── tests/          # Unit and regression tests
├── examples/       # Minimal runnable example
├── tools/          # Benchmark tool (SPMV_BUILD_BENCHMARKS=ON)
├── docs/           # GitHub Pages site
├── CHANGELOG.md    # Single project changelog
└── CMakeLists.txt
```

## Documentation

Documentation is published at **https://aicl-lab.github.io/gpu-spmv/**.

| Page | Purpose |
|:-----|:--------|
| [Quick Start](https://aicl-lab.github.io/gpu-spmv/en/quickstart) | Installation and build flow |
| [API Reference](https://aicl-lab.github.io/gpu-spmv/en/api/spmv) | Core public API |
| [Architecture](https://aicl-lab.github.io/gpu-spmv/en/architecture/overview) | Data flow and kernel selection |
| [Performance Guide](https://aicl-lab.github.io/gpu-spmv/en/performance/optimization-guide) | Practical tuning notes |
| [Examples](https://aicl-lab.github.io/gpu-spmv/en/examples/basic-spmv) | End-to-end usage |

Version history is kept only in the root [CHANGELOG.md](CHANGELOG.md).

## Contributing

Keep changes boring and verifiable:

1. Make the smallest change that improves the core library.
2. Preserve RAII resource handling; do not introduce raw `cudaMalloc` / `cudaFree`.
3. Run the existing build and test commands.
4. Update the relevant documentation when behavior changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the short contribution workflow.

## License

MIT License. See [LICENSE](LICENSE).
