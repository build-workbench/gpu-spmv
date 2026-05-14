---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">GPU</div>
    <div>
      <span class="home-title">GPU SpMV</span>
      <span class="home-subtitle">High-Performance Sparse Matrix-Vector Multiplication</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./whitepaper/">Whitepaper</a>
    <a href="https://github.com/LessUp/gpu-spmv">GitHub</a>
    <a href="../zh/">中文</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    GPU SpMV is a production-grade CUDA library for high-performance sparse matrix-vector multiplication (SpMV). With 4 optimized kernels and intelligent selection algorithms, it achieves 70%+ theoretical bandwidth utilization on modern NVIDIA GPUs. Supports CSR and ELL formats with comprehensive API and 100+ test case coverage.
  </div>
  <div class="home-stats">
    <span><strong>70%+</strong> Bandwidth</span>
    <span><strong>4</strong> Kernels</span>
    <span><strong>100+</strong> Tests</span>
  </div>
</div>

## Core Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">🚀 Extreme Performance</div>
    <div class="feature-card-desc">
      Merge Path for perfect load balancing, ELL for fully coalesced access, auto kernel selection for optimal performance.
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/performance" class="feature-tag">Performance</a>
      <a href="./performance/benchmarks" class="feature-tag">Benchmarks</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📊 Multiple Formats</div>
    <div class="feature-card-desc">
      CSR for general sparse matrices, ELL for GPU-friendly layout, automatic format conversion, seamless GPU/CPU switching.
    </div>
    <div class="feature-tags">
      <a href="./api/csr-matrix" class="feature-tag">CSR API</a>
      <a href="./api/ell-matrix" class="feature-tag">ELL API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🎯 Production Quality</div>
    <div class="feature-card-desc">
      RAII resource management, semantic error codes, cross-platform support, comprehensive test coverage.
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview" class="feature-tag">Architecture</a>
      <a href="./api/spmv" class="feature-tag">API Reference</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📐 Spec-Driven Development</div>
    <div class="feature-card-desc">
      OpenSpec specification-driven, traceable design decisions, automated change management, documentation as code.
    </div>
    <div class="feature-tags">
      <a href="./architecture/spec-driven" class="feature-tag">Workflow</a>
      <a href="./whitepaper/philosophy" class="feature-tag">Philosophy</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🔬 Academic Rigor</div>
    <div class="feature-card-desc">
      Complete academic citation support, BibTeX format, related paper references, reproducible benchmarks.
    </div>
    <div class="feature-tags">
      <a href="./references" class="feature-tag">References</a>
      <a href="./citation" class="feature-tag">Citation</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📚 Complete Documentation</div>
    <div class="feature-card-desc">
      Technical whitepaper, API reference, architecture design, performance guide, bilingual support.
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/" class="feature-tag">Whitepaper</a>
      <a href="./api/spmv" class="feature-tag">API</a>
    </div>
  </div>
</div>

<div class="quick-start">
  <div class="quick-start-title">Quick Start</div>
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
