-- -layout : default title : Home nav_order : 1 permalink : / index.en lang : en-- -

        <!-- ═══════════════════════════════════════════════════════════════ GPU SpMV -
         Product Showcase Homepage(English)
     ═══════════════════════════════════════════════════════════════ -->

        <!--Hero Section--><div class = "hero-section"><div class = "hero-content">
        <h1 class = "hero-title"> GPU SpMV</ h1>
        <p class = "hero-subtitle"> High - Performance CUDA Sparse Matrix -
        Vector Multiplication</ p><p class = "hero-tagline"> Intelligent Kernel Selection · 70 %
            +Bandwidth · Production Ready</ p>

            <!--Badges--><div class = "hero-badges"> <
    span class
    = "badge badge-primary" >
      70 % +Bandwidth</ span><span class = "badge badge-secondary"> CUDA 11.0 + </ span>
          <span class = "badge badge-success"> MIT License</ span>
          <span class = "badge badge-info"> C++ 17 <
      / span >
      </ div>

      <!--CTA Buttons--><div class = "hero-buttons">
      <a href = "quickstart.en" class = "btn btn-primary btn-lg"><span>🚀</ span> Quick Start</ a>
      <a href = "https://github.com/LessUp/gpu-spmv" class = "btn btn-secondary btn-lg" target =
           "_blank">
      <span>📦</ span> GitHub</ a><a href = "performance.en" class = "btn btn-tertiary btn-lg">
      <span>📊</ span> Benchmarks</ a></ div></ div>

      <!--Quick Code Preview--><div class = "hero-code"><div class = "code-window">
      <div class = "code-header"><span class = "dot red"></ span><span class = "dot yellow"></ span>
      <span class = "dot green"></ span><span class = "code-title"> example.cpp</ span></ div> {
    % highlight cpp %
}
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
    printf("Bandwidth: %.1f%%\n", result.bandwidth_utilization * 100);
}
{% endhighlight %}
    </div>
  </div>
</div>

---

<!-- Features Section -->
## Key Features
{: .section-title }

<div class="feature-grid">
  <div class="feature-card perf">
    <div class="feature-icon">🚀</div>
    <h3>Extreme Performance</h3>
    <ul>
      <li>4 optimized kernels with intelligent selection</li>
      <li>Up to <strong>70%+</strong> theoretical bandwidth</li>
      <li>Merge Path for perfect load balancing</li>
      <li>ELL format with coalesced memory access</li>
    </ul>
  </div>
  
  <div class="feature-card format">
    <div class="feature-icon">📊</div>
    <h3>Multi-Format Support</h3>
    <ul>
      <li><strong>CSR</strong> - General sparse matrices</li>
      <li><strong>ELL</strong> - High-performance uniform matrices</li>
      <li>Automatic format conversion</li>
      <li>Seamless GPU/CPU switching</li>
    </ul>
  </div>
  
  <div class="feature-card quality">
    <div class="feature-icon">🎯</div>
    <h3>Production Quality</h3>
    <ul>
      <li>RAII resource management (CudaBuffer)</li>
      <li>Semantic error codes (SpMVError)</li>
      <li>Cross-platform (Linux/Windows)</li>
      <li>100+ test cases coverage</li>
    </ul>
  </div>
</div>

---

<!-- Performance Section -->
## Performance
{: .section-title }

<div class="perf-showcase">
  <div class="perf-table-wrapper">
    <table class="perf-table">
      <thead>
        <tr>
          <th>Matrix Size</th>
          <th>Non-zeros</th>
          <th>Kernel</th>
          <th>Bandwidth</th>
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
    <p class="perf-note">Benchmarks: NVIDIA RTX 3090 (Ampere, 936 GB/s)</p>
  </div>
  
  <div class="perf-cta">
    <p>View detailed benchmarks and optimization guides</p>
    <a href="performance.en" class="btn btn-primary">View Performance →</a>
  </div>
</div>

---

<!-- Quick Start Section -->
## Quick Start
{: .section-title }

<div class="quickstart-section">
  <div class="install-command">
    <h4>Installation</h4>
    <div class="code-block">
      <pre><code>git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset release && cmake --build --preset release</code></pre>
      <button class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
    </div>
  </div>
  
  <div class="quick-links">
    <a href="quickstart.en" class="quick-link">
      <span>📚</span>
      <strong>Full Installation Guide</strong>
      <small>Requirements & detailed steps</small>
    </a>
    <a href="examples.en" class="quick-link">
      <span>📝</span>
      <strong>Code Examples</strong>
      <small>7 complete examples</small>
    </a>
    <a href="api.en" class="quick-link">
      <span>📖</span>
      <strong>API Reference</strong>
      <small>Complete interface docs</small>
    </a>
  </div>
</div>

---

<!-- Architecture Preview -->
## Architecture
{: .section-title
}

```
┌─────────────────────────────────────────────────────────┐
│ Application Layer                     │
│ PageRank  │ Iterative  │ Graph NNs  │ Scientific     │
├─────────────────────────────────────────────────────────┤
│ API Layer                          │
│ spmv_csr  │ spmv_ell │ benchmark │ pagerank      │
├─────────────────────────────────────────────────────────┤
│ Kernel Layer                        │
│ Scalar CSR │ Vector CSR │ Merge Path │ ELL Kernel    │
├─────────────────────────────────────────────────────────┤
│ Storage Layer                        │
│ CSR Matrix      │ ELL Matrix         │
└─────────────────────────────────────────────────────────┘
```

    <p align = "center">
    <a href = "architecture.en" class = "btn"> View Architecture →</ a></ p>

    -- -

    <!--Use Cases-->##Use Cases{ :.section - title}

    < div class
    = "usecase-grid" > <div class = "usecase-item"><strong>🕸️ Graph Algorithms</ strong>
      <span> PageRank,
    shortest path,
    community detection</ span></ div><div class = "usecase-item">
    <strong>🔬 Scientific Computing</ strong><span> Finite element analysis,
    CFD</ span></ div><div class = "usecase-item"><strong>🤖 Machine Learning</ strong>
    <span> Sparse neural networks,
    recommendations</ span></ div><div class = "usecase-item"><strong>📊 Data Analytics</ strong>
    <span> Matrix factorization,
    eigenvalue computation</ span></ div></ div>

    -- -

    <!--Footer CTA--><div class = "footer-cta">
    <h2> Get Started with GPU SpMV</ h2>
    <p> Join thousands of developers accelerating sparse matrix computations</ p>
    <div class = "footer-buttons">
    <a href = "quickstart.en" class = "btn btn-primary btn-lg"> Quick Start →</ a>
    <a href = "https://github.com/LessUp/gpu-spmv" class = "btn btn-secondary btn-lg" target =
         "_blank">
    <span>⭐</ span> Star on GitHub</ a></ div></ div>

    -- -

    <div class = "text-center text-small text-grey-dk-300" style = "margin-top: 3rem;"><p>
    <a href = "/">🇨🇳 查看中文版</ a></ p><p> GPU SpMV &copy;
2024 - 2026 LessUp |
    <a href = "https://github.com/LessUp/gpu-spmv/blob/master/LICENSE"> MIT License</ a></ p></ div>

    <script> function copyToClipboard(btn) {
    const code = btn.previousElementSibling.innerText;
    navigator.clipboard.writeText(code).then(() = > {
        btn.textContent = 'Copied!';
        setTimeout(() = > btn.textContent = 'Copy', 2000);
    });
}
</script>
