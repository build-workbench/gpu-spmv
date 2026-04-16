---
layout: default
title: Home
nav_order: 1
has_children: false
permalink: /index.en
---

# GPU SpMV Documentation
{: .fs-9 }

High-Performance CUDA Sparse Matrix-Vector Multiplication Library
{: .fs-6 .fw-300 }

[Get Started](installation.en){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View Examples](examples.en){: .btn .btn-green .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub](https://github.com/LessUp/gpu-spmv){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## ✨ Features

### Multi-Format Support

| Format | Description | GPU Efficiency |
|:-------|:------------|:-------------:|
| **CSR** | Compressed Sparse Row - General purpose | ★★★☆☆ |
| **ELL** | ELLPACK - Uniform row lengths | ★★★★★ |

### Intelligent Kernel Selection

```
Matrix Analysis
       │
       ├── ELL format → ELL Kernel (coalesced access)
       │
       └── CSR format
              │
              ├── avg_nnz < 4 → Scalar CSR (1 thread/row)
              │
              └── avg_nnz >= 4
                     │
                     ├── skewness < 10 → Vector CSR (warp/row)
                     │
                     └── skewness >= 10 → Merge Path (balanced)
```

### Production Quality

- 🎯 **RAII Resource Management** — `CudaBuffer` automatic lifecycle
- 🔍 **Semantic Error Codes** — `SpMVError` clear tracking
- 🖥️ **Cross-Platform** — Windows / Linux
- 🔧 **CMake Presets** — One-click builds
- ✅ **Full Test Coverage** — Google Test

---

## 🚀 Quick Start

### Requirements

- CUDA Toolkit 11.0+ / 12.0+
- CMake 3.18+
- C++17 compiler
- NVIDIA GPU (CC 7.0+)

### Installation

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset release
cmake --build --preset release
```

### 30-Second Example

```cpp
#include <spmv/spmv.h>

CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, data, 3, 3);
csr_to_gpu(csr);

SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| [📦 Installation](installation.en) | System requirements, dependencies |
| [🏗️ Architecture](architecture.en) | System design, algorithms |
| [📚 API Reference](api.en) | Complete API documentation |
| [📝 Examples](examples.en) | Basic to advanced examples |
| [🚀 Performance](performance.en) | Optimization strategies |
| [📋 Changelog](changelog.en) | Version history |

---

## 📊 Performance

| Matrix Size | Non-zeros | Kernel | Bandwidth |
|:-----------:|:---------:|:-------|:---------:|
| 10K × 10K | 500K | Vector CSR | ~70% |
| 100K × 100K | 5M | Merge Path | ~65% |
| 1M × 1M | 50M | Merge Path | ~60% |

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p>GPU SpMV is licensed under <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">MIT License</a></p>
  <p>Copyright &copy; 2024-2025 LessUp</p>
</div>
