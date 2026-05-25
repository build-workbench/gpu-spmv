# Contributing

Thank you for your interest in GPU SpMV.

## Development Setup

```bash
git clone https://github.com/AICL-Lab/gpu-spmv.git
cd gpu-spmv
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

CPU-only environments:

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda
ctest --test-dir build-no-cuda --output-on-failure
```

On Linux, use the official CUDA presets so the build always uses the system GCC/G++ host toolchain:

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

## Contribution Rules

1. Keep changes focused on the core SpMV library.
2. Preserve RAII resource management and explicit error handling.
3. Run the existing tests.
4. Update the relevant docs when behavior changes.

## Code Style

- 4-space indentation
- 100-character line width
- Google C++ style
- `clang-format` for modified files

## Documentation

- Chinese pages live in `docs/zh/`
- English pages live in `docs/en/`
- Mermaid is available for diagrams

## Getting Help

- Open an [Issue](https://github.com/AICL-Lab/gpu-spmv/issues)
- Read the existing docs
