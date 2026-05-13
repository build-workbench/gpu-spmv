# Changelog

All notable changes to GPU SpMV are documented here.

## [1.0.0] - 2026-04-01

### Added
- Complete SpMV implementation with 4 optimized kernels
- CSR and ELL sparse matrix formats
- Automatic kernel selection based on matrix statistics
- PageRank algorithm implementation
- Comprehensive benchmark suite
- RAII memory management with CudaBuffer
- Semantic error codes (SpMVError)
- OpenSpec specification-driven development

### Performance
- 70%+ memory bandwidth utilization on RTX 3090
- Merge Path kernel for load-balanced computation
- ELL kernel for coalesced memory access
- Texture cache support for large vectors

### Documentation
- Bilingual documentation (Chinese/English)
- Complete API reference
- Architecture design documentation
- Academic references

## [0.9.0] - 2025-03-10

### Added
- PageRank application layer
- Top-K node extraction utility

## [0.8.0] - 2025-03-05

### Added
- Benchmark framework with statistics
- JSON export for benchmark results

## [0.7.0] - 2025-03-01

### Added
- Automatic kernel selection (`spmv_auto_config`)
- Configurable selection thresholds

## [0.6.0] - 2025-02-20

### Added
- Merge Path kernel for skewed matrices
- Matrix statistics computation

### Performance
- Improved load balancing for irregular matrices

## [0.5.0] - 2025-02-10

### Added
- ELL matrix format
- ELL kernel for uniform matrices
- CSR to ELL conversion

## [0.4.0] - 2025-01-15

### Added
- CSR matrix format
- Scalar CSR kernel
- Vector CSR kernel
- Basic SpMV computation

## [0.1.0] - 2024-12-01

### Added
- Initial project structure
- CMake build system
- Google Test integration
