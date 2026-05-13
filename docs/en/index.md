---
layout: home

hero:
  name: GPU SpMV
  text: High-Performance Sparse Matrix-Vector Multiplication
  tagline: 4 Optimized Kernels · 70%+ Bandwidth · Production Ready · Spec-Driven
  actions:
    - theme: brand
      text: Quick Start
      link: /en/quickstart
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/gpu-spmv
    - theme: alt
      text: Benchmarks
      link: /en/performance/benchmarks

features:
  - icon: 🚀
    title: Extreme Performance
    details: |
      <ul>
        <li>4 optimized kernels with intelligent selection</li>
        <li>Up to <strong>70%+</strong> theoretical bandwidth</li>
        <li>Merge Path for perfect load balancing</li>
        <li>ELL format with coalesced memory access</li>
      </ul>
  - icon: 📊
    title: Multi-Format Support
    details: |
      <ul>
        <li><strong>CSR</strong> — General sparse matrices</li>
        <li><strong>ELL</strong> — High-performance uniform matrices</li>
        <li>Automatic format conversion</li>
        <li>Seamless GPU/CPU switching</li>
      </ul>
  - icon: 🎯
    title: Production Quality
    details: |
      <ul>
        <li>RAII resource management (CudaBuffer)</li>
        <li>Semantic error codes (SpMVError)</li>
        <li>Cross-platform (Linux/Windows)</li>
        <li>100+ test cases coverage</li>
      </ul>
  - icon: 📐
    title: Spec-Driven Development
    details: |
      <ul>
        <li>OpenSpec specification-driven</li>
        <li>Traceable design decisions</li>
        <li>Automated change management</li>
        <li>Documentation as code</li>
      </ul>
---

<div class="sp-home-extra">

## Code Preview

<div class="sp-code-window">
  <div class="sp-code-header">
    <span class="sp-dot sp-dot-red"></span>
    <span class="sp-dot sp-dot-yellow"></span>
    <span class="sp-dot sp-dot-green"></span>
    <span class="sp-code-title">example.cpp</span>
  </div>

```cpp
#include <spmv/spmv.h>

int main() {
    // Create sparse matrix
    CSRMatrix* csr = csr_create(10000, 10000, 500000);
    csr_from_dense(csr, data, 10000, 10000);
    csr_to_gpu(csr);

    // Auto-select optimal kernel and execute
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

    // 70%+ bandwidth utilization
    printf("Bandwidth: %.1f%%\n",
           result.bandwidth_utilization * 100);
}
```
</div>

## Performance

| Matrix Size | Non-zeros | Kernel | Bandwidth |
|:-----------:|:---------:|:-------|:---------:|
| 10K × 10K | 500K | Vector CSR | **70.2%** |
| 100K × 100K | 5M | Merge Path | **71.5%** |
| 1M × 1M | 50M | Merge Path | **70.8%** |

<p class="sp-perf-note">Benchmarks: NVIDIA RTX 3090 (Ampere, 936 GB/s)</p>

## Architecture

```mermaid
graph TB
    subgraph Application["Application"]
        PR[PageRank]
        IS[Iterative Solver]
        GNN[Graph NN]
        SC[Scientific]
    end

    subgraph API["API"]
        spmv_csr[spmv_csr]
        spmv_ell[spmv_ell]
        benchmark[benchmark]
        pagerank[pagerank]
    end

    subgraph Kernel["Kernel"]
        Scalar["Scalar CSR"]
        Vector["Vector CSR"]
        Merge["Merge Path"]
        ELL["ELL Kernel"]
    end

    subgraph Storage["Storage"]
        CSR_M["CSR Matrix"]
        ELL_M["ELL Matrix"]
    end

    Application --> API
    API --> Kernel
    Kernel --> Storage
```

## Use Cases

- 🕸️ **Graph Algorithms** — PageRank, shortest path, community detection
- 🔬 **Scientific Computing** — Finite element analysis, CFD
- 🤖 **Machine Learning** — Sparse neural networks, recommendations
- 📊 **Data Analytics** — Matrix factorization, eigenvalue computation

</div>

<style>
.sp-home-extra {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 var(--spacing-lg);
}

.sp-code-window {
  border-radius: var(--radius-lg);
  overflow: hidden;
  border: 1px solid var(--vp-c-border);
  margin: var(--spacing-lg) 0;
}

.sp-code-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--vp-code-block-bg);
  border-bottom: 1px solid var(--vp-code-block-border);
}

.sp-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.sp-dot-red { background: #FF5F57; }
.sp-dot-yellow { background: #FEBC2E; }
.sp-dot-green { background: #28C840; }

.sp-code-title {
  margin-left: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.sp-perf-note {
  text-align: center;
  font-size: 13px;
  color: var(--vp-c-text-3);
}
</style>
