# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SpMVConfig::enable_timing`: set to `false` to enqueue kernels without CUDA events or synchronization, so SpMV calls can be pipelined asynchronously (default `true` keeps the blocking, metered behavior).
- `spmv_result_error()` typed accessor for `SpMVResult::error_code`, and `spmv_auto_config_ell()` for API symmetry.
- `csr_read_matrix_market()` (`spmv/market_io.h`): Matrix Market coordinate reader (real/integer/pattern, general/symmetric; duplicates summed).
- Runnable `examples/basic_spmv.cpp` (built by default, works in CPU-only builds) and a synthetic benchmark tool `tools/spmv_bench.cu` behind `SPMV_BUILD_BENCHMARKS=ON`.
- CI `cuda-test` job that runs the full GPU test suite on a self-hosted runner when the `CUDA_RUNNER_LABEL` repository variable is set.
- Serialization format version 2: the integrity checksum now also covers the `values` array. Version 1 files remain readable.

### Fixed
- `csr_read_matrix_market` no longer crashes (`std::length_error` from `vector::reserve`) on headers claiming more entries than the file can contain; such files are rejected with `FILE_IO`, and entry counts are capped at `INT_MAX / 2` (symmetric expansion bound).
- Installed package now exports its include directory: `include(GNUInstallDirs)` ran after the target definition, so `gpu_spmv::spmv` lost `INTERFACE_INCLUDE_DIRECTORIES` and `find_package()` consumers could not compile.
- Creation and conversion functions (`csr_create`, `csr_from_dense`, `ell_create`, `ell_from_dense`, `ell_from_csr`) now use nothrow allocations and return the documented `nullptr` / `OUT_OF_MEMORY` on failure instead of letting `std::bad_alloc` escape a C-style API.
- `csr_create` / `csr_from_dense` reject `rows == INT_MAX`, where the `rows + 1` allocation would overflow.
- `ell_from_csr` allocates before releasing the previous arrays, so a failed allocation leaves the matrix intact (matching `ell_from_dense` / `csr_from_dense`).
- `CudaBuffer::resize` keeps the original buffer intact when the device-to-device copy fails, instead of leaving a null pointer with a stale size.
- Kernel grid size math (`scalar` / `vector` / `merge path` / `ELL`) uses an overflow-safe ceil division for extreme matrix sizes.
- README minimal example now compiles (`CudaBuffer` exposes `get()`, not `data()`).
- `SpMVResult::bandwidth_gb_s` is now derived from the reported `elapsed_ms` instead of a separate host-side timer, so the result fields are mutually consistent.
- `spmv_csr`/`spmv_ell` zero-work paths (empty matrix / zero-length vector) now synchronize like the kernel paths when timing is enabled, making blocking behavior independent of matrix contents.
- Merge Path grid is now partitioned by `nnz` instead of `num_rows`, restoring parallelism for the highly skewed matrices this kernel exists for.
- ELL kernel uses 64-bit index math, so matrices with `num_rows * max_nnz_per_row > INT_MAX` no longer overflow.
- `csr_deserialize`/`ell_deserialize` reject headers that claim more payload than the file contains (before allocating) and guard the `rows + 1` allocation against integer overflow.
- `csr_get_element` returns correct values for CSR matrices whose column indices are not sorted within each row.
- The L2 persisting access-policy hint is restored on exit, so `spmv_csr`/`spmv_ell` no longer leave side effects on the caller's stream.

### Changed
- Reduced the repository to the core CSR / ELL SpMV library and removed repository-specific AI governance files.
- Simplified contributor workflow, GitHub templates, and GitHub Pages content to match the smaller core scope.
- Added dedicated Linux CUDA presets backed by system GCC/G++ and fail-fast guidance for Conda host compilers.
- `spmv_cpu_csr`/`spmv_cpu_ell` now return an `int` error code instead of silently ignoring invalid input.
- CI `build-cpu` job now installs the package and builds a `find_package()` consumer as a packaging smoke test.
- `CudaBuffer::resize` preserves existing elements (like `std::vector::resize`) instead of discarding them.
- Internal `select_kernel()` no longer takes an unused `num_cols` parameter.

### Removed
- OpenSpec specifications, Claude / Copilot repository instruction files, and local skill configuration.
- Built-in PageRank and benchmark modules, their tests, and their documentation pages.
- GitHub Pages changelog mirroring; the root `CHANGELOG.md` is now the only changelog.

## [1.0.0] - 2025-04-16

### 🎉 First Stable Release

This is the first stable release of GPU SpMV, featuring complete CSR and ELL format support, four optimized CUDA kernels with automatic selection, and production-ready engineering quality.

### ✨ Added

#### Core Features
- **CSR (Compressed Sparse Row)** sparse matrix format with full operations
- **ELL (ELLPACK)** sparse matrix format with column-major GPU-optimized storage
- **Four CUDA Kernels**: Scalar CSR, Vector CSR, Merge Path, ELL Kernel
- **Automatic kernel selection** based on matrix statistics (avg_nnz, skewness)
- **Texture cache support** with `SpMVExecutionContext` for object reuse
- **RAII resource management**: `CudaBuffer<T>`, `CudaTimer`, `ScopedTexture`
- **Semantic error codes**: `SpMVError` enum with descriptive error messages

#### Performance & Benchmarking
- Bandwidth metrics calculation with GPU peak bandwidth detection
- Comprehensive benchmarking framework with warmup runs and statistical analysis
- GPU vs CPU performance comparison with speedup metrics
- JSON export for benchmark results

#### Applications
- **PageRank algorithm** with GPU-accelerated iterative computation
- Configurable damping factor and convergence tolerance
- Top-K node ranking extraction

#### Engineering Quality
- CMake Presets for easy Debug/Release builds
- CPU-only configuration option for development environments
- Cross-platform support (Windows/Linux)
- Complete Google Test test suite with property-based testing
- GitHub Actions CI/CD with format checking
- Doxygen-compatible documentation

#### Documentation
- Full documentation site at https://aicl-lab.github.io/gpu-spmv/
- Bilingual README (English and Chinese)
- API reference, performance guide, and code examples
- Architecture documentation and design decision records

### 🔒 Security
- Integer overflow protection in size calculations
- Memory bounds checking in matrix operations

### 🚀 Performance
- ELL Column-major storage for fully coalesced memory access
- Warp-level shuffle reduction avoiding shared memory bank conflicts
- Merge Path algorithm for perfect load balancing on irregular matrices
- Automatic texture cache for large input vectors (>10000 elements)

## [0.1.0] - 2025-03-01

### 🚀 Initial Release

- Basic project structure
- Initial CSR matrix implementation
- Simple SpMV GPU kernel
- CMake build configuration

---

## Version History

| Version | Date | Status | Highlights |
|:-------:|:----:|:------:|:-----------|
| [1.0.0] | 2025-04-16 | Stable | First stable release with complete feature set |
| [0.1.0] | 2025-03-01 | Archived | Initial prototype |

---

## Migration Guide

### Upgrading to 1.0.0

No breaking changes from pre-release versions. The API is now stable.

#### Recommended Updates

1. **Use named constants** instead of magic numbers:
   ```cpp
   // Before
   config.block_size = 256;
   config.use_texture = (cols > 10000);

   // After (recommended)
   config.block_size = spmv::DEFAULT_BLOCK_SIZE;
   config.use_texture = (cols > spmv::TEXTURE_CACHE_THRESHOLD_COLS);
   ```

2. **Use `SpMVExecutionContext`** for texture object reuse:
   ```cpp
   // Before: Texture created/destroyed each call
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols);
   }

   // After: Reuse texture across calls
   SpMVExecutionContext context;
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols, &context);
   }
   ```

3. **Check error codes** consistently:
   ```cpp
   SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);
   if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
       std::cerr << "Error: " << spmv_error_string(
           static_cast<SpMVError>(result.error_code)) << std::endl;
   }
   ```

---

## Future Roadmap

### Planned for 1.1.0

- [ ] COO (Coordinate) format support
- [ ] Hybrid CSR/ELL format
- [ ] Multi-GPU support
- [ ] Batched SpMV operations
- [ ] Double precision support

### Under Consideration

- [ ] BFloat16 precision support
- [ ] Automatic format selection tuning
- [ ] Integration with cuSPARSE for comparison
- [ ] Python bindings

---

[1.0.0]: https://github.com/AICL-Lab/gpu-spmv/releases/tag/v1.0.0
[0.1.0]: https://github.com/AICL-Lab/gpu-spmv/tree/7d6dd0c
