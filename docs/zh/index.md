---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">GPU</div>
    <div>
      <span class="home-title">GPU SpMV</span>
      <span class="home-subtitle">技术白皮书</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="./whitepaper/">白皮书</a>
    <a href="https://github.com/LessUp/gpu-spmv">GitHub</a>
    <a href="../en/">English</a>
  </div>
</div>

<div class="home-hero-tech">
  <h1>生产级 CUDA 稀疏矩阵向量乘法</h1>
  <p class="hero-tagline">
    高性能稀疏矩阵向量乘法（SpMV），在现代 NVIDIA GPU 上实现 <strong>70%+ 理论内存带宽利用率</strong>。
    4 个自适应内核、智能选择算法、完整 API。
  </p>
  <div class="hero-actions">
    <a href="./whitepaper/" class="primary">阅读白皮书</a>
    <a href="./quickstart" class="secondary">快速开始</a>
  </div>
</div>

<div class="home-metrics">
  <div class="home-metric">
    <div class="home-metric-value">70%+</div>
    <div class="home-metric-label">带宽利用率</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">4</div>
    <div class="home-metric-label">自适应内核</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">CSR+ELL</div>
    <div class="home-metric-label">稀疏格式</div>
  </div>
  <div class="home-metric">
    <div class="home-metric-value">100+</div>
    <div class="home-metric-label">测试用例</div>
  </div>
</div>

## 架构概览

<div class="home-architecture">

```mermaid
flowchart LR
    Input[稀疏矩阵] --> Analysis[矩阵分析]
    Analysis --> Decision{自动选择}
    Decision -->|avg_nnz < 4| Scalar[Scalar CSR]
    Decision -->|均匀行| Vector[Vector CSR]
    Decision -->|高偏斜| Merge[Merge Path]
    Decision -->|列主序| ELL[ELL 内核]
    Scalar --> GPU[GPU 执行]
    Vector --> GPU
    Merge --> GPU
    ELL --> GPU
    GPU --> Result[结果向量]
```

</div>

## 技术特性

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">内核选择策略</div>
    <div class="feature-card-desc">
      基于矩阵特征自动选择最优内核：平均非零元数、行长度偏斜度。
    </div>
    <div class="feature-tags">
      <a href="./architecture/kernel-selection" class="feature-tag">详细说明</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Merge Path 算法</div>
    <div class="feature-card-desc">
      针对不规则稀疏模式的完美负载均衡，O(nnz + m) 工作分解。
    </div>
    <div class="feature-tags">
      <a href="./whitepaper/philosophy" class="feature-tag">设计哲学</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">生产级质量</div>
    <div class="feature-card-desc">
      RAII 资源管理、语义化错误码、CudaBuffer 抽象、跨平台支持。
    </div>
    <div class="feature-tags">
      <a href="./api/spmv" class="feature-tag">API</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">Spec-Driven 开发</div>
    <div class="feature-card-desc">
      OpenSpec 规范驱动工作流，设计决策可追溯，文档即代码。
    </div>
    <div class="feature-tags">
      <a href="./architecture/spec-driven" class="feature-tag">工作流</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">学术严谨</div>
    <div class="feature-card-desc">
      完整的学术引用支持、BibTeX 格式、相关论文参考。
    </div>
    <div class="feature-tags">
      <a href="./citation" class="feature-tag">引用格式</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">快速开始</div>
    <div class="feature-card-desc">
      <code>git clone https://github.com/LessUp/gpu-spmv.git</code>
    </div>
    <div class="feature-tags">
      <a href="./quickstart" class="feature-tag">详细指南</a>
    </div>
  </div>
</div>
