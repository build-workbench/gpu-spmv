---
layout: default
title: Quick Start
nav_order: 2
permalink: /quickstart.en
lang: en
---

<p align="right">
  <a href="quickstart">🇨🇳 简体中文</a>
</p>

# Quick Start
{: .no_toc }

Get GPU SpMV installed and running in 5 minutes.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Requirements

| Component | Minimum | Recommended |
|:----------|:-------:|:-----------:|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Compiler | GCC 7+ / MSVC 2019+ | GCC 11+ / MSVC 2022+ |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |
| GPU Memory | 4 GB | 8 GB+ |

### Check CUDA Installation

```bash
nvcc --version
nvidia-smi
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
```

### 2. Build Project

Using CMake Presets (recommended):

```bash
# Release build
cmake --preset release
cmake --build --preset release
```

Or traditional way:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. Run Tests

```bash
# Run all tests
ctest --preset default

# Or run directly
./build-release/spmv_tests
```

---

## First Program

Create `first_spmv.cpp`:

```cpp
#include <spmv/spmv.h>
#include <cstdio>

int main() {
    // Create 3x3 sparse matrix: [1 0 2; 0 3 4; 0 0 5]
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    
    CSRMatrix* csr = csr_create(3, 3, 5);
    csr_from_dense(csr, dense, 3, 3);
    csr_to_gpu(csr);
    
    // Prepare input vector x = [1, 1, 1]
    float h_x[] = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    cudaMemcpy(d_x.data(), h_x, 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Execute SpMV: y = A * x
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.data(), d_y.data(), &config, 3);
    
    if (result.error == SpMVError::SUCCESS) {
        printf("Success! Time: %.3f ms\n", result.time_ms);
        
        // Read result
        float h_y[3];
        cudaMemcpy(h_y, d_y.data(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Result: [%.0f, %.0f, %.0f]\n", h_y[0], h_y[1], h_y[2]);
        // Output: [3, 7, 5]
    }
    
    csr_destroy(csr);
    return 0;
}
```

### Compile and Run

```bash
# Compile
nvcc -o first_spmv first_spmv.cpp \
    -I./include \
    -L./build-release -lgpu_spmv \
    -lcudart

# Run
./first_spmv
```

---

## Next Steps

- **[API Reference](api.en)** - Complete interface documentation
- **[Examples](examples.en)** - More code examples
- **[Performance](performance.en)** - Optimization guides

---

## FAQ

### Q: Build fails, cannot find CUDA?

Ensure CUDA is properly installed and environment variables are set:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: Tests fail?

Check if GPU is available:

```bash
nvidia-smi
```

If no GPU, use CPU-only test mode:

```bash
cmake --preset minimal
```

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>Problems? <a href="https://github.com/LessUp/gpu-spmv/issues">Submit Issue</a></p>
</div>
