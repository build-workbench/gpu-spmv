---
layout: default
title: Home
nav_order: 1
permalink: /
lang: zh
---

<!-- ═══════════════════════════════════════════════════════════════
     GPU SpMV - Product Showcase Homepage (Chinese)
     ═══════════════════════════════════════════════════════════════ -->

<!-- Hero Section -->
<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">GPU SpMV</h1>
    <p class="hero-subtitle">基于 CUDA 的高性能稀疏矩阵向量乘法库</p>
    <p class="hero-tagline">智能 Kernel 调度 · 70%+ 带宽利用率 · 生产级可靠</p>
    
    <!-- Badges -->
    <div class="hero-badges">
      <span class="badge badge-primary">70%+ Bandwidth</span>
      <span class="badge badge-secondary">CUDA 11.0+</span>
      <span class="badge badge-success">MIT License</span>
      <span class="badge badge-info">C++17</span>
    </div>
    
    <!-- CTA Buttons -->
    <div class="hero-buttons">
      <a href="quickstart" class="btn btn-primary btn-lg">
        <span>🚀</span> 快速开始
      </a>
      <a href="https://github.com/LessUp/gpu-spmv" class="btn btn-secondary btn-lg" target="_blank">
        <span>📦</span> GitHub
      </a>
      <a href="performance" class="btn btn-tertiary btn-lg">
        <span>📊</span> 性能对比
      </a>
    </div>
  </div>
  
  <!-- Quick Code Preview -->
  <div class="hero-code">
    <div class="code-window">
      <div class="code-header">
        <span class="dot red"></span>
        <span class="dot yellow"></span>
        <span class="dot green"></span>
        <span class="code-title">example.cpp</span>
      </div>
{% highlight cpp %}
#include <spmv/spmv.h>

int main() {
    // 创建稀疏矩阵
    CSRMatrix* csr = csr_create(10000, 10000, 500000);
    csr_from_dense(csr, data, 10000, 10000);
    csr_to_gpu(csr);
    
    // 智能 Kernel 选择并执行
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
    
    // 70%+ 带宽利用率
    printf("Bandwidth: %.1f%%\n", 
           result.bandwidth_utilization * 100);
}
{% endhighlight %}
    </div>
  </div>
</div>

---

<!-- Features Section -->
## 核心特性
{: .section-title }

<div class="feature-grid">
  <div class="feature-card perf">
    <div class="feature-icon">🚀</div>
    <h3>极致性能</h3>
    <ul>
      <li>4 种优化 Kernel 智能调度</li>
      <li>高达 <strong>70%+</strong> 理论带宽利用</li>
      <li>Merge Path 完美负载均衡</li>
      <li>ELL 格式完全合并访存</li>
    </ul>
  </div>
  
  <div class="feature-card format">
    <div class="feature-icon">📊</div>
    <h3>多格式支持</h3>
    <ul>
      <li><strong>CSR</strong> - 通用稀疏矩阵</li>
      <li><strong>ELL</strong> - 高性能均匀矩阵</li>
      <li>格式间自动转换</li>
      <li>GPU/CPU 无缝切换</li>
    </ul>
  </div>
  
  <div class="feature-card quality">
    <div class="feature-icon">🎯</div>
    <h3>生产级质量</h3>
    <ul>
      <li>RAII 资源管理（CudaBuffer）</li>
      <li>语义化错误码（SpMVError）</li>
      <li>跨平台支持（Linux/Windows）</li>
      <li>100+ 测试用例覆盖</li>
    </ul>
  </div>
</div>

---

<!-- Performance Section -->
## 性能表现
{: .section-title }

<div class="perf-showcase">
  <div class="perf-table-wrapper">
    <table class="perf-table">
      <thead>
        <tr>
          <th>矩阵规模</th>
          <th>非零元素</th>
          <th>Kernel</th>
          <th>带宽利用率</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>10K × 10K</td>
          <td>500K</td>
          <td>Vector CSR</td>
          <td><span class="perf-high">70.2%</span></td>
        </tr>
        <tr>
          <td>100K × 100K</td>
          <td>5M</td>
          <td>Merge Path</td>
          <td><span class="perf-high">71.5%</span></td>
        </tr>
        <tr>
          <td>1M × 1M</td>
          <td>50M</td>
          <td>Merge Path</td>
          <td><span class="perf-high">70.8%</span></td>
        </tr>
      </tbody>
    </table>
    <p class="perf-note">测试环境：NVIDIA RTX 3090 (Ampere, 936 GB/s)</p>
  </div>
  
  <div class="perf-cta">
    <p>查看详细基准测试和优化指南</p>
    <a href="performance" class="btn btn-primary">查看性能详情 →</a>
  </div>
</div>

---

<!-- Quick Start Section -->
## 快速开始
{: .section-title }

<div class="quickstart-section">
  <div class="install-command">
    <h4>安装</h4>
    <div class="code-block">
      <pre><code>git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset release && cmake --build --preset release</code></pre>
      <button class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
    </div>
  </div>
  
  <div class="quick-links">
    <a href="quickstart" class="quick-link">
      <span>📚</span>
      <strong>完整安装指南</strong>
      <small>系统要求、详细步骤</small>
    </a>
    <a href="examples" class="quick-link">
      <span>📝</span>
      <strong>示例代码</strong>
      <small>7 个完整示例</small>
    </a>
    <a href="api" class="quick-link">
      <span>📖</span>
      <strong>API 文档</strong>
      <small>完整接口参考</small>
    </a>
  </div>
</div>

---

<!-- Architecture Preview -->
## 架构设计
{: .section-title }

```
┌─────────────────────────────────────────────────────────┐
│                      应用层                              │
│   PageRank  │  迭代求解器  │  图神经网络  │  科学计算    │
├─────────────────────────────────────────────────────────┤
│                      API 层                              │
│   spmv_csr  │  spmv_ell  │  benchmark  │  pagerank    │
├─────────────────────────────────────────────────────────┤
│                     Kernel 层                           │
│  Scalar CSR │ Vector CSR │ Merge Path │  ELL Kernel   │
├─────────────────────────────────────────────────────────┤
│                      存储层                              │
│              CSR Matrix      │      ELL Matrix        │
└─────────────────────────────────────────────────────────┘
```

<p align="center"><a href="architecture" class="btn">查看架构详情 →</a></p>

---

<!-- Use Cases -->
## 应用场景
{: .section-title }

<div class="usecase-grid">
  <div class="usecase-item">
    <strong>🕸️ 图算法</strong>
    <span>PageRank、最短路径、社区发现</span>
  </div>
  <div class="usecase-item">
    <strong>🔬 科学计算</strong>
    <span>有限元分析、计算流体力学</span>
  </div>
  <div class="usecase-item">
    <strong>🤖 机器学习</strong>
    <span>稀疏神经网络、推荐系统</span>
  </div>
  <div class="usecase-item">
    <strong>📊 数据分析</strong>
    <span>矩阵分解、特征值计算</span>
  </div>
</div>

---

<!-- Footer CTA -->
<div class="footer-cta">
  <h2>开始使用 GPU SpMV</h2>
  <p>加入数千开发者，加速您的稀疏矩阵计算</p>
  <div class="footer-buttons">
    <a href="quickstart" class="btn btn-primary btn-lg">快速开始 →</a>
    <a href="https://github.com/LessUp/gpu-spmv" class="btn btn-secondary btn-lg" target="_blank">
      <span>⭐</span> Star on GitHub
    </a>
  </div>
</div>

---

<div class="text-center text-small text-grey-dk-300" style="margin-top: 3rem;">
  <p><a href="index.en">🇺🇸 View in English</a></p>
  <p>GPU SpMV &copy; 2024-2026 LessUp | <a href="https://github.com/LessUp/gpu-spmv/blob/main/LICENSE">MIT License</a></p>
</div>

<script>
function copyToClipboard(btn) {
  const code = btn.previousElementSibling.innerText;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 2000);
  });
}
</script>
