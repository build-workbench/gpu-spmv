---
layout: home
---

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
  <h1>Production-Grade CUDA Sparse Matrix-Vector Multiplication</h1>
  <p class="hero-tagline">
    High-performance SpMV achieving <strong>70%+ theoretical memory bandwidth</strong> on modern NVIDIA GPUs.
    4 adaptive kernels, intelligent selection algorithm, comprehensive API.
  </p>
  <div class="hero-actions">
    <a href="./whitepaper/" class="primary">Read the Whitepaper</a>
    <a href="./quickstart" class="secondary">Quick Start</a>
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

## Technical Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">Kernel Selection Strategy</div>
    <div class="feature-card-desc">
      Automatic kernel selection based on matrix characteristics: avg_nnz, row length skewness.
    </div>
    <div class="feature-tags">
      <a href="./architecture/kernel-selection" class="feature-tag">Details</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Merge Path Algorithm</div>
    <div class="feature-card-desc">
      Perfect load balancing for irregular sparsity patterns. O(nnz + m) work decomposition.
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/philosophy" class="feature-tag">Philosophy</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Production Quality</div>
    <div class="feature-card-desc">
      RAII resource management, semantic error codes, CudaBuffer abstraction, cross-platform.
    </div>
    <div class="feature-tags">
      <a href="./api/spmv" class="feature-tag">API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Spec-Driven Development</div>
    <div class="feature-card-desc">
      OpenSpec specification-driven workflow. Design decisions traceable, documentation as code.
    </div>
    <div class="feature-tags">
      <a href="./architecture/spec-driven" class="feature-tag">Workflow</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Academic Rigor</div>
    <div class="feature-card-desc">
      Complete academic citation support, BibTeX format, related paper references.
    </div>
    <div class="feature-tags">
      <a href="./citation" class="feature-tag">Citation</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Quick Start</div>
    <div class="feature-card-desc">
      <code>git clone https://github.com/LessUp/gpu-spmv.git</code>
    </div>
    <div class="feature-tags">
      <a href="./quickstart" class="feature-tag">Guide</a>
    </div>
  </div>
</div>
