---
layout: home
---

<script setup>
import PerformanceChart from '../.vitepress/components/PerformanceChart.vue'
import TrendChart from '../.vitepress/components/TrendChart.vue'
import KernelSelector from '../.vitepress/components/KernelSelector.vue'
import ComparisonTable from '../.vitepress/components/ComparisonTable.vue'
</script>

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">GPU</div>
    <div>
      <span class="home-title">GPU SpMV</span>
      <span class="home-subtitle">Technical Whitepaper</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./whitepaper/">Whitepaper</a>
    <a href="https://github.com/LessUp/gpu-spmv">GitHub</a>
    <a href="../zh/">中文</a>
  </div>
</div>

<div class="home-hero-tech">
  <h1>Production-Grade CUDA SpMV</h1>
  <p class="hero-tagline">
    High-performance sparse matrix-vector multiplication achieving <strong>70%+ theoretical memory bandwidth</strong> on modern NVIDIA GPUs. 4 adaptive kernels, intelligent selection algorithm, comprehensive API.
  </p>
  <div class="hero-actions">
    <a href="./whitepaper/" class="primary">Read the Whitepaper</a>
    <a href="./performance/benchmarks" class="secondary">View Benchmarks</a>
  </div>
</div>

<div class="home-metrics">
  <div class="home-metric">
    <div class="home-metric-value">70%+</div>
    <div class="home-metric-label">Bandwidth Utilization</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">4</div>
    <div class="home-metric-label">Adaptive Kernels</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">CSR+ELL</div>
    <div class="home-metric-label">Sparse Formats</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">100+</div>
    <div class="home-metric-label">Test Cases</div>
  </div>
</div>

## Interactive Performance Dashboard

<div class="home-dashboard">
  <PerformanceChart title="Kernel Performance by Matrix Type" />
  <TrendChart title="Performance vs Matrix Size" />
</div>

## Architecture Overview

<div class="home-architecture">

```mermaid
flowchart LR
    Input[Sparse Matrix] --> Analysis[Matrix Analysis]
    Analysis --> Decision{Auto Select}
    Decision -->|avg_nnz < 4| Scalar[Scalar CSR]
    Decision -->|uniform rows| Vector[Vector CSR]
    Decision -->|high skew| Merge[Merge Path]
    Decision -->|column-major| ELL[ELL Kernel]
    Scalar --> GPU[GPU Execution]
    Vector --> GPU
    Merge --> GPU
    ELL --> GPU
    GPU --> Result[Result Vector]
```

</div>

### Kernel Selection Explorer

<KernelSelector />

## Competitive Positioning

<ComparisonTable lang="en" />

## Technical Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">Kernel Selection Strategy</div>
    <div class="feature-card-desc">
      Automatic kernel selection based on matrix characteristics: avg_nnz, row length skewness, and distribution pattern.
    </div>
    <div class="feature-tags">
      <a href="./architecture/kernel-selection" class="feature-tag">Details</a>
      <a href="./whitepaper/performance" class="feature-tag">Analysis</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Memory Layout Optimization</div>
    <div class="feature-card-desc">
      ELL column-major layout for fully coalesced memory access. CSR row-major for general sparse patterns.
    </div>
    <div class="feature-tags">
      <a href="./architecture/memory-layout" class="feature-tag">Memory</a>
      <a href="./api/ell-matrix" class="feature-tag">ELL API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Merge Path Algorithm</div>
    <div class="feature-card-desc">
      Perfect load balancing for irregular sparsity patterns. O(nnz + m) work decomposition across thread blocks.
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/philosophy" class="feature-tag">Philosophy</a>
      <a href="./architecture/overview" class="feature-tag">Architecture</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Production Quality</div>
    <div class="feature-card-desc">
      RAII resource management, semantic error codes (SpMVError), CudaBuffer abstraction, cross-platform support.
    </div>
    <div class="feature-tags">
      <a href="./api/spmv" class="feature-tag">API</a>
      <a href="./architecture/overview" class="feature-tag">Design</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Spec-Driven Development</div>
    <div class="feature-card-desc">
      OpenSpec specification-driven workflow. Design decisions are traceable, changes are managed, documentation is code.
    </div>
    <div class="feature-tags">
      <a href="./architecture/spec-driven" class="feature-tag">Workflow</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Academic Rigor</div>
    <div class="feature-card-desc">
      Complete academic citation support, BibTeX format, related paper references, reproducible benchmarks.
    </div>
    <div class="feature-tags">
      <a href="./references" class="feature-tag">References</a>
      <a href="./citation" class="feature-tag">Citation</a>
    </div>
  </div>
</div>

## Quick Start

<div class="quick-start">
  <div class="quick-start-title">Build from Source</div>
  <div class="quick-start-content">
    <div class="command-block">
      <code>git clone https://github.com/LessUp/gpu-spmv.git</code>
    </div>
    <div class="command-block">
      <code>cmake -S . -B build && cmake --build build</code>
    </div>
    See the <a href="./quickstart">Quick Start Guide</a> for more details.
  </div>
</div>
