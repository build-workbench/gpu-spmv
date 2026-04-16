# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-04-16

### Added

#### Core Features

- **CSR (Compressed Sparse Row) sparse matrix format**
  - Create, destroy, serialize, deserialize operations
  - Dense matrix conversion and element access
  - GPU memory management with automatic synchronization
  - Statistics computation (avg/max/min nnz per row, skewness)

- **ELL (ELLPACK) sparse matrix format**
  - Column-major storage for optimal GPU memory coalescing
  - CSR-to-ELL conversion
  - Padding detection and storage efficiency metrics

- **Four optimized CUDA kernels**
  - **Scalar CSR**: One thread per row, optimal for very sparse matrices
  - **Vector CSR**: One warp (32 threads) per row, efficient for uniform distributions
  - **Merge Path**: Perfect load balancing for highly irregular matrices
  - **ELL Kernel**: Column-major access for uniform row lengths

- **Automatic kernel selection**
  - Matrix statistics analysis (avg_nnz_per_row, skewness)
  - Decision tree based kernel recommendation
  - Configurable through `SpMVConfig`

- **Texture cache support**
  - Optional texture cache for input vector reads
  - `SpMVExecutionContext` for texture object reuse
  - Automatic threshold-based enabling (num_cols > 10000)

#### Performance & Benchmarking

- **Bandwidth metrics**
  - Peak bandwidth caching via `std::call_once`
  - Achieved bandwidth calculation
  - Efficiency percentage reporting

- **Benchmarking framework**
  - Configurable warmup and run counts
  - Statistical analysis (avg, min, max, stddev)
  - GPU vs CPU comparison with speedup metrics
  - JSON export for results

#### Applications

- **PageRank algorithm**
  - GPU-accelerated iterative computation
  - Configurable damping factor and tolerance
  - Top-K node ranking extraction
  - Convergence tracking

#### Engineering Quality

- **RAII resource management**
  - `CudaBuffer<T>`: GPU memory with memset/fill/resize
  - `CudaTimer`: CUDA Events-based precise timing
  - `ScopedTexture`: Texture object lifecycle management

- **Semantic error codes**
  - `SpMVError` enum with descriptive values
  - `CUDA_CHECK_MALLOC` / `CUDA_CHECK_MEMCPY` macros
  - `spmv_error_string()` for human-readable messages
  - `CudaException` for RAII error propagation

- **Build system**
  - CMake Presets (default, release, minimal)
  - CPU-only configuration option (`DSPMV_REQUIRE_CUDA=OFF`)
  - Project version metadata (1.0.0)
  - Installation rules for library and headers

- **Code quality**
  - `.clang-format` (Google-based, 4-space indent, 100 col)
  - `.editorconfig` for editor consistency
  - Cross-platform test path handling

- **Testing**
  - Property-based tests with 100 random iterations
  - CSR/ELL conversion correctness
  - SpMV computation verification vs CPU reference
  - Dimension validation tests
  - Kernel selector validity tests
  - Bandwidth metric validation
  - PageRank invariant checks

#### Documentation

- **GitHub Pages site** at https://lessup.github.io/gpu-spmv/
  - Landing page with feature highlights
  - Complete API reference
  - Performance optimization guide
  - Code examples collection

- **Bilingual README**
  - English (README.md) as primary
  - Chinese (README.zh-CN.md) translation

- **CI/CD**
  - GitHub Actions workflow
  - clang-format validation
  - CPU-only build verification

### Security

- **Integer overflow protection**
  - Size calculation validation in CSR operations
  - Size calculation validation in ELL operations
  - Prevents undefined behavior with extreme inputs

### Performance

- **Memory bandwidth optimization**
  - Column-major storage for ELL format
  - Warp-level shuffle reduction (no bank conflicts)
  - Texture cache for large input vectors

- **Load balancing**
  - Merge Path algorithm for highly skewed matrices
  - Automatic kernel selection based on matrix structure

## [0.1.0] - 2025-03-01

### Added

- Initial project structure
- Basic CSR matrix implementation
- Simple SpMV GPU kernel
- CMake build configuration

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-04-16 | First stable release with full feature set |
| 0.1.0 | 2025-03-01 | Initial prototype |

---

## Migration Guide

### Upgrading to 1.0.0

No breaking changes from pre-release versions. The API is now stable.

#### Recommended Updates

1. **Use named constants** instead of magic numbers:
   ```cpp
   // Old
   config.block_size = 256;
   config.use_texture = (cols > 10000);

   // New (recommended)
   config.block_size = spmv::DEFAULT_BLOCK_SIZE;
   config.use_texture = (cols > spmv::TEXTURE_CACHE_THRESHOLD_COLS);
   ```

2. **Use `SpMVExecutionContext`** for texture object reuse:
   ```cpp
   // Old: Texture object created/destroyed each call
   for (int i = 0; i < iterations; i++) {
       config.use_texture = true;
       spmv_csr(csr, d_x, d_y, &config, cols);
   }

   // New: Texture object reused across calls
   SpMVExecutionContext context;
   config.use_texture = true;
   for (int i = 0; i < iterations; i++) {
       spmv_csr(csr, d_x, d_y, &config, cols, &context);
   }
   ```

3. **Check error codes** consistently:
   ```cpp
   SpMVResult result = spmv_csr(csr, d_x, d_y, &config, cols);
   if (result.error_code != static_cast<int>(SpMVError::SUCCESS)) {
       fprintf(stderr, "Error: %s\n", spmv_error_string(
           static_cast<SpMVError>(result.error_code)));
   }
   ```

---

## Future Roadmap

### Planned for 1.1.0

- [ ] COO (Coordinate) format support
- [ ] Hybrid CSR/ELL format
- [ ] Multi-GPU support
- [ ] Batched SpMV operations

### Under Consideration

- [ ] BFloat16 precision support
- [ ] Sparse matrix storage format auto-tuning
- [ ] Integration with cuSPARSE for comparison
- [ ] Python bindings

---

[1.0.0]: https://github.com/LessUp/gpu-spmv/releases/tag/v1.0.0
[0.1.0]: https://github.com/LessUp/gpu-spmv/tree/7d6dd0c
