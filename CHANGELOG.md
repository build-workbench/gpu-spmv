# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Reduced the repository to the core CSR / ELL SpMV library and removed repository-specific AI governance files.
- Simplified contributor workflow, GitHub templates, and GitHub Pages content to match the smaller core scope.
- Added dedicated Linux CUDA presets backed by system GCC/G++ and fail-fast guidance for Conda host compilers.

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
