# Frequently Asked Questions

## Installation & Configuration

### CUDA Version Requirements?

GPU SpMV requires the following CUDA versions:

| CUDA Version | Supported GPU Architectures |
|:------------:|:---------------------------|
| 11.0+ | Volta (SM 7.0), Turing (SM 7.5) |
| 11.1+ | Ampere (SM 8.0, 8.6) |
| 11.8+ | Ada Lovelace (SM 8.9) |
| 12.0+ | Hopper (SM 9.0) |

**Recommended**: CUDA 12.0+ for best performance and compatibility.

::: tip No GPU Environment
Use `-DSPMV_REQUIRE_CUDA=OFF` to build CPU-only version without GPU:
```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda
```
:::

### Supported Operating Systems?

- **Linux**: Ubuntu 20.04+, CentOS 7+, Debian 10+
- **Windows**: Windows 10/11 with Visual Studio 2019+
- **macOS**: Not supported (no NVIDIA GPU)

### How to Verify Installation?

Run the test suite:

```bash
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

All tests passing indicates successful installation.

---

## Matrix Formats

### CSR vs ELL Format - Which to Choose?

| Format | Use Case | Performance Characteristics |
|:-------|:---------|:----------------------------|
| **CSR** | General sparse matrices, irregular non-zero distribution | Suitable for most cases, auto kernel selection |
| **ELL** | Similar non-zero count per row | Fully coalesced access, highest bandwidth utilization |

**Recommendation**: Use CSR by default, convert to ELL for uniform matrices for better performance.

### How to Convert Matrix Format?

```cpp
#include <spmv/spmv.h>

// Convert from CSR to ELL
CSRMatrix* csr = csr_create(rows, cols, nnz);
// ... fill CSR ...

ELLMatrix* ell = ell_create(rows, cols, max_nnz_per_row);
ell_from_csr(ell, csr);  // Automatic conversion
```

---

## Performance Optimization

### Why is My Performance Below 70%?

Possible causes and solutions:

1. **Matrix too small**
   - Issue: GPU not fully utilized
   - Solution: Matrix size should be > 10K × 10K

2. **Extremely uneven non-zero distribution**
   - Issue: Load imbalance
   - Solution: Merge Path kernel handles this automatically, or try adjusting matrix structure

3. **Older GPU architecture**
   - Issue: Missing modern GPU features
   - Solution: Use GPU with Compute Capability 7.0+

4. **Memory bandwidth limitation**
   - Issue: Other processes using GPU memory
   - Solution: Ensure sufficient GPU memory, close other GPU applications

### How to Select Optimal Kernel?

Use `spmv_auto_config()` for automatic selection:

```cpp
SpMVConfig config = spmv_auto_config(csr);
// Automatically selects optimal kernel based on matrix characteristics
```

Selection strategy:
- `avg_nnz_per_row < 4` → Scalar CSR
- `skewness < 10` → Vector CSR
- `skewness >= 10` → Merge Path

### How to Reuse Configuration for Batch Operations?

```cpp
// Compute configuration once
SpMVConfig config = spmv_auto_config(csr);

// Reuse configuration for multiple SpMV operations
for (int i = 0; i < iterations; i++) {
    spmv_csr(csr, x[i], y[i], &config, n);
}
```

---

## Comparison with Other Libraries

### How Does It Compare to cuSPARSE?

| Feature | GPU SpMV | cuSPARSE |
|:--------|:--------:|:--------:|
| Open Source | ✅ | ❌ |
| Auto Kernel Selection | ✅ | ❌ |
| Merge Path Algorithm | ✅ | ❌ |
| ELL Format Support | ✅ | ✅ |
| Irregular Matrix Performance | **Better** | Average |
| Uniform Matrix Performance | Similar | Similar |

### Comparison with Other Open Source Libraries?

| Library | Stars | Features |
|:--------|:-----:|:---------|
| **GPU SpMV** | - | Auto selection, Merge Path, complete docs |
| [Ginkgo](https://github.com/ginkgo-project/ginkgo) | 597 | Multi-backend, performance portability |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) | 300+ | Performance portability, multi-platform |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | N/A | Official, multi-format |

---

## Troubleshooting

### Compilation Error: CUDA not found

Ensure CUDA is installed correctly:

```bash
# Check CUDA version
nvcc --version

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda
```

### Runtime Error: invalid device ordinal

GPU device index error:

```cpp
// Check available GPU count
int device_count;
cudaGetDeviceCount(&device_count);

// Set correct device
cudaSetDevice(0);  // Use first GPU
```

### Unstable Benchmark Results

Ensure:
1. GPU temperature is normal (avoid thermal throttling)
2. No other GPU processes interfering
3. Warmup before testing

```cpp
// Warmup
for (int i = 0; i < 10; i++) {
    spmv_csr(csr, x, y, &config, n);
}

// Actual benchmark
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 100; i++) {
    spmv_csr(csr, x, y, &config, n);
}
auto end = std::chrono::high_resolution_clock::now();
```

---

## More Questions?

If the above doesn't answer your question:

1. Check [API Reference](/en/api/spmv) for detailed usage
2. Check [Optimization Guide](/en/performance/optimization-guide) for performance tips
3. Ask on [GitHub Issues](https://github.com/AICL-Lab/gpu-spmv/issues)
