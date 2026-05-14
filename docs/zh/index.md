---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">GPU</div>
    <div>
      <span class="home-title">GPU SpMV</span>
      <span class="home-subtitle">高性能稀疏矩阵向量乘法</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./whitepaper/">技术白皮书</a>
    <a href="https://github.com/LessUp/gpu-spmv">GitHub</a>
    <a href="../en/">English</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    GPU SpMV 是一个生产级 CUDA 库，实现了高性能稀疏矩阵向量乘法（SpMV）。通过 4 种优化内核和智能选择算法，在现代 NVIDIA GPU 上达到 70%+ 理论带宽利用率。支持 CSR 和 ELL 格式，提供完整的 API 和 100+ 测试用例覆盖。
  </div>
  <div class="home-stats">
    <span><strong>70%+</strong> 带宽利用</span>
    <span><strong>4</strong> 种内核</span>
    <span><strong>100+</strong> 测试用例</span>
  </div>
</div>

## 核心特性

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">🚀 极致性能</div>
    <div class="feature-card-desc">
      Merge Path 完美负载均衡，ELL 完全合并访存，自动内核选择确保最优性能。
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/performance" class="feature-tag">性能分析</a>
      <a href="./performance/benchmarks" class="feature-tag">基准测试</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📊 多格式支持</div>
    <div class="feature-card-desc">
      CSR 通用稀疏矩阵，ELL GPU 友好格式，格式间自动转换，GPU/CPU 无缝切换。
    </div>
    <div class="feature-tags">
      <a href="./api/csr-matrix" class="feature-tag">CSR API</a>
      <a href="./api/ell-matrix" class="feature-tag">ELL API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🎯 生产级质量</div>
    <div class="feature-card-desc">
      RAII 资源管理，语义化错误码，跨平台支持，完善的测试覆盖。
    </div>
    <div class="feature-tags">
      <a href="./architecture/overview" class="feature-tag">架构设计</a>
      <a href="./api/spmv" class="feature-tag">API 参考</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📐 Spec-Driven 开发</div>
    <div class="feature-card-desc">
      OpenSpec 规范驱动，可追溯设计决策，自动变更管理，文档即代码。
    </div>
    <div class="feature-tags">
      <a href="./architecture/spec-driven" class="feature-tag">开发流程</a>
      <a href="./whitepaper/philosophy" class="feature-tag">设计哲学</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">🔬 学术严谨</div>
    <div class="feature-card-desc">
      完整的学术引用支持，BibTeX 格式，相关论文参考，可复现基准测试。
    </div>
    <div class="feature-tags">
      <a href="./references" class="feature-tag">学术参考</a>
      <a href="./citation" class="feature-tag">引用格式</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">📚 完整文档</div>
    <div class="feature-card-desc">
      技术白皮书、API 参考、架构设计、性能指南，中英文双语支持。
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/" class="feature-tag">白皮书</a>
      <a href="./api/spmv" class="feature-tag">API</a>
    </div>
  </div>
</div>

<div class="quick-start">
  <div class="quick-start-title">快速开始</div>
  <div class="quick-start-content">
    <div class="command-block">
      <code>git clone https://github.com/LessUp/gpu-spmv.git</code>
    </div>
    <div class="command-block">
      <code>cmake -S . -B build && cmake --build build</code>
    </div>
    查看 <a href="./quickstart">快速开始指南</a> 了解更多。
  </div>
</div>
