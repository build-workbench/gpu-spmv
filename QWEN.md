# QWEN.md - GPU SpMV Project Context

## Project Overview

**GPU SpMV** is a production-ready C++/CUDA library that accelerates **Sparse Matrix-Vector Multiplication (SpMV)** on NVIDIA GPUs. It delivers up to **70%+ of theoretical memory bandwidth** through intelligent kernel selection and optimized memory access patterns.

### Key Features
- **4 Optimized CUDA Kernels**: Scalar CSR, Vector CSR, Merge Path, ELL
- **2 Sparse Matrix Formats**: CSR (general-purpose) and ELL (uniform row lengths)
- **Automatic Kernel Selection**: Based on matrix characteristics (sparsity, skewness)
- **Production-Grade Quality**: RAII resource management, semantic error codes, full test coverage
- **Cross-Platform**: Windows and Linux support

### Primary Use Cases
- Graph algorithms (PageRank, shortest path)
- Scientific computing (finite element analysis, CFD)
- Machine learning (sparse neural networks, recommendation systems)
- Data analytics (matrix decomposition, eigenvalue computation)

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Languages** | C++17, CUDA |
| **Build System** | CMake 3.18+ with presets |
| **Testing** | Google Test (v1.14.0) |
| **Code Style** | clang-format (Google style) |
| **CI/CD** | GitHub Actions |
| **Documentation** | Jekyll + Just the Docs (GitHub Pages) |

### Supported GPU Architectures
- Volta (7.0, 7.5)
- Turing (7.5)
- Ampere (8.0, 8.6)
- Ada Lovelace (8.9)
- Hopper (9.0)

---

## Project Structure

```
gpu-spmv/
├── include/spmv/          # Public headers (10 files)
│   ├── common.h           # Error codes (SpMVError), CUDA macros
│   ├── cuda_buffer.h      # RAII GPU memory (CudaBuffer<T>)
│   ├── csr_matrix.h       # CSR sparse matrix operations
│   ├── ell_matrix.h       # ELL sparse matrix operations
│   ├── spmv.h             # Main SpMV interface & kernel selection
│   ├── bandwidth.h        # Bandwidth metrics utilities
│   ├── benchmark.h        # Performance benchmarking framework
│   ├── pagerank.h         # PageRank algorithm interface
│   ├── matrix_wrapper.h   # Format conversion utilities
│   └── test_utils.h       # Testing utilities
├── src/                   # Source implementations (7 files)
│   ├── csr_matrix.cpp     # CSR operations
│   ├── ell_matrix.cpp     # ELL operations
│   ├── spmv_cpu.cpp       # CPU-side SpMV logic
│   ├── spmv_kernels.cu    # CUDA kernels (4 variants)
│   ├── bandwidth.cpp      # Bandwidth calculations
│   ├── benchmark.cu       # Benchmark implementation
│   └── pagerank.cu        # PageRank algorithm
├── tests/                 # Google Test suite (8 files)
├── benchmarks/            # Performance benchmarks
├── specs/                 # SDD specifications (Single Source of Truth)
│   ├── product/           # Product requirements (PRDs)
│   ├── rfc/               # Technical design decisions
│   ├── api/               # API specifications
│   ├── db/                # Database specs (placeholder)
│   └── testing/           # BDD test specifications
├── docs/                  # Documentation (GitHub Pages)
│   ├── setup/             # Installation guides
│   ├── tutorials/         # API, examples, performance
│   ├── architecture/      # Architecture docs
│   └── assets/            # Static assets
└── .github/workflows/     # CI/CD pipelines
```

---

## Building and Running

### Prerequisites
- **CUDA Toolkit**: 11.0+ (12.0+ recommended)
- **CMake**: 3.18+
- **C++ Compiler**: C++17 support
- **NVIDIA GPU**: Compute Capability 7.0+

### Build Commands

```bash
# Configure and build (Debug)
cmake --preset default && cmake --build --preset default

# Configure and build (Release - recommended for performance)
cmake --preset release && cmake --build --preset release

# Configure and build (Minimal - sm_80 only, faster build)
cmake --preset minimal && cmake --build --preset minimal

# CPU-only configure (no CUDA device needed)
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
```

### Running Tests

```bash
# Run all tests (using ctest preset)
ctest --preset default

# Or run test executable directly
./build/spmv_tests

# Run specific test category
./build/spmv_tests --gtest_filter="CSR*"
./build/spmv_tests --gtest_filter="ELL*"

# Property tests with random matrices (10 iterations)
./build/spmv_tests --gtest_repeat=10
```

### Running Benchmarks

```bash
# Run performance benchmarks
./build-release/spmv_benchmark

# Example output:
# GPU: NVIDIA GeForce RTX 3090
# Matrix: 100000x100000, NNZ: 5000000
# Avg time: 23.5 ms | Bandwidth: 69.8 GB/s (71.5% of peak)
```

### Installation

```bash
# Install to system
cmake --install build-release

# Uninstall
sudo xargs rm < build/install_manifest.txt
```

---

## Development Conventions

### Spec-Driven Development (SDD)

**This project strictly follows Spec-Driven Development.** The `/specs/` directory is the **Single Source of Truth** for all implementation decisions.

#### AI Agent Workflow (MANDATORY)

When developing features, fixing bugs, or modifying functionality:

1. **Review Specs First**:
   - `/specs/product/` - Feature requirements
   - `/specs/rfc/` - Technical design decisions
   - `/specs/api/` - API interface definitions
   - `/specs/testing/` - Test requirements

2. **Spec-First Updates**:
   - Propose spec changes BEFORE writing code
   - Wait for user confirmation on spec changes
   - Do NOT proceed to coding until specs are approved

3. **Implementation**:
   - Follow specs 100% (variable naming, API paths, data types, status codes)
   - NO gold-plating (only implement what's in specs)

4. **Test Against Specs**:
   - Use `/specs/testing/` for test cases
   - Property-based tests: minimum 100 iterations
   - Cover all boundary conditions in specs

### Code Style

- **Formatter**: clang-format (Google style)
  - 4-space indentation
  - 100 character line limit
  - Pointer alignment: Left
- **Include Order** (enforced by `.clang-format`):
  1. `"spmv/..."` (project headers)
  2. `<cuda...>` (CUDA headers)
  3. `<...>` (standard library)
  4. Everything else
- **Namespace**: `spmv::`
- **Error Handling**: `SpMVError` enum + `CUDA_CHECK*` macros
- **Resource Management**: RAII patterns (`CudaBuffer<T>`, `SpMVExecutionContext`)
  - Never use raw `cudaMalloc`/`cudaFree` in new code

### Commit Messages

Use **Conventional Commits** format:
```
<type>: <description>

# Types: feat, fix, perf, build, refactor, docs, test, ci, chore
# Examples:
feat: add COO format support
fix: resolve memory leak in pagerank
perf: optimize merge path kernel
build: update cmake preset configuration
```

### Testing Practices

- **Framework**: Google Test
- **Property-Based Testing**: 100 iterations with random matrices
- **Test File Naming**: `*_test.cpp` or `*_tests.cu`
- **Coverage Areas**:
  - CSR/ELL format conversion correctness
  - SpMV computation correctness (vs CPU reference)
  - Dimension validation
  - Kernel selection logic
  - Bandwidth metrics
  - PageRank invariants

---

## Key Architecture Decisions

### Kernel Selection Strategy

| Matrix Pattern | Kernel | Strategy | Performance |
|---------------|--------|----------|-------------|
| Very sparse (avg_nnz < 4) | Scalar CSR | 1 thread/row | ★★★☆☆ |
| Uniform (skewness < 10) | Vector CSR | 1 warp/row | ★★★★☆ |
| Skewed (skewness ≥ 10) | Merge Path | Perfect load balancing | ★★★★★ |
| ELL format | ELL Kernel | Coalesced access | ★★★★★ |

### Resource Management

All GPU resources use RAII patterns:
```cpp
// Automatic lifecycle management
CudaBuffer<float> d_x(1000);  // Allocates on construction
// Automatically freed on destruction (scope exit)
```

### Error Handling

```cpp
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
if (result.error != SpMVError::SUCCESS) {
    fprintf(stderr, "Error: %s\n", spmv_error_string(result.error));
}
```

---

## Common Pitfalls

### GPU/CI Environment
- **CI has no GPU** - Tests requiring CUDA devices will fail in CI
- `benchmarks/main.cu` and `pagerank.cu` exit early if no CUDA device found
- Use CPU-only tests for CI validation

### Build System
- **No Makefile** - All builds via CMake presets
- Three presets: `default` (Debug), `release` (Release), `minimal` (sm_80 only)
- CPU-only configure: `cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF`

### Code Formatting
- CI uses **clang-format-18** for format checking
- Format code before committing:
  ```bash
  find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
    | xargs clang-format -i
  ```

---

## Documentation

### GitHub Pages Site
- **URL**: https://lessup.github.io/gpu-spmv/
- **Source**: `docs/` directory
- **Framework**: Jekyll + Just the Docs theme
- **Languages**: Chinese + English

### Key Documentation Pages
- [Installation Guide](https://lessup.github.io/gpu-spmv/installation)
- [API Reference](https://lessup.github.io/gpu-spmv/api)
- [Examples](https://lessup.github.io/gpu-spmv/examples)
- [Performance Guide](https://lessup.github.io/gpu-spmv/performance)
- [Architecture](https://lessup.github.io/gpu-spmv/architecture)
- [Changelog](https://lessup.github.io/gpu-spmv/changelog)

---

## Quick Reference

```bash
# Workflow: Review Specs → Update Specs → Get Approval → Implement → Test

# Build
cmake --preset default && cmake --build --preset default

# Test
ctest --preset default

# Format code
find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  | xargs clang-format -i

# Specs location
/specs/{product, rfc, api, db, testing}/

# Docs location
/docs/{setup, tutorials, architecture, assets}/
```

---

## Useful Links

- **GitHub Repository**: https://github.com/LessUp/gpu-spmv
- **Documentation**: https://lessup.github.io/gpu-spmv/
- **Algorithm Paper**: [Merge-based Parallel SpMV](https://research.nvidia.com/publication/merge-based-parallel-sparse-matrix-vector-multiplication) by Merrill & Garland (NVIDIA)
