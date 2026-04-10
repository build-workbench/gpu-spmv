# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GPU-accelerated Sparse Matrix-Vector Multiplication (SpMV) library in C++/CUDA. Supports CSR and ELL formats with multiple kernel strategies and auto-selection. Includes PageRank and benchmarking.

## Build & Test

```bash
# Configure + build
cmake --preset default && cmake --build --preset default   # Debug
cmake --preset release && cmake --build --preset release   # Release

# CPU-only configure (no CUDA device needed)
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF

# Test
ctest --preset default
./build/spmv_tests              # direct
```

No Makefile — all via CMake presets. Three presets: `default` (Debug), `release`, `minimal` (sm_80 only).

## Code Style

- clang-format enforced in CI (`.clang-format` based on Google style, 4-space indent, 100 col limit)
- Include order: `"spmv/"` first, then `<cuda*`, then `<`, then everything else
- Namespace: `spmv::`
- Error handling: `SpMVError` enum, `CUDA_CHECK`/`CUDA_CHECK_MALLOC`/`CUDA_CHECK_MEMCPY` macros, RAII via `CudaBuffer<T>`

## Conventions

- Commit messages: Conventional Commits (`feat:`, `fix:`, `perf:`, `build:`, `refactor:`, etc.)
- Tests: Google Test, property-based tests use 100 iterations with random matrices
- Design specs live in `.kiro/specs/spmv-gpu/` (requirements, design, tasks)

## Key Gotchas

- CI has no GPU — tests requiring CUDA devices will fail in CI
- `benchmarks/main.cu` and `pagerank.cu` exit early with error if no CUDA device found
- The `minimal` preset hardcodes sm_80; other presets let CMake auto-detect architectures
