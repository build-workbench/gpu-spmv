---
layout: home

hero:
  name: GPU SpMV
  text: 高性能稀疏矩阵向量乘法
  tagline: 4 优化内核 · 70%+ 带宽利用率 · 生产级可靠 · Spec-Driven
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/quickstart
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/gpu-spmv
    - theme: alt
      text: 性能对比
      link: /zh/performance/benchmarks

features:
  - icon: 🚀
    title: 极致性能
    details: |
      <ul>
        <li>4 种优化 Kernel 智能调度</li>
        <li>高达 <strong>70%+</strong> 理论带宽利用</li>
        <li>Merge Path 完美负载均衡</li>
        <li>ELL 格式完全合并访存</li>
      </ul>
  - icon: 📊
    title: 多格式支持
    details: |
      <ul>
        <li><strong>CSR</strong> — 通用稀疏矩阵</li>
        <li><strong>ELL</strong> — 高性能均匀矩阵</li>
        <li>格式间自动转换</li>
        <li>GPU/CPU 无缝切换</li>
      </ul>
  - icon: 🎯
    title: 生产级质量
    details: |
      <ul>
        <li>RAII 资源管理 (CudaBuffer)</li>
        <li>语义化错误码 (SpMVError)</li>
        <li>跨平台支持 (Linux/Windows)</li>
        <li>100+ 测试用例覆盖</li>
      </ul>
  - icon: 📐
    title: Spec-Driven 开发
    details: |
      <ul>
        <li>OpenSpec 规范驱动</li>
        <li>可追溯设计决策</li>
        <li>自动变更管理</li>
        <li>文档即代码</li>
      </ul>
---

<div class="sp-home-extra">

## 代码预览

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

## 性能表现

| 矩阵规模 | 非零元素 | Kernel | 带宽利用率 |
|:--------:|:--------:|:-------|:----------:|
| 10K × 10K | 500K | Vector CSR | **70.2%** |
| 100K × 100K | 5M | Merge Path | **71.5%** |
| 1M × 1M | 50M | Merge Path | **70.8%** |

<p class="sp-perf-note">测试环境：NVIDIA RTX 3090 (Ampere, 936 GB/s)</p>

## 架构设计

```mermaid
graph TB
    subgraph Application["应用层"]
        PR[PageRank]
        IS[迭代求解器]
        GNN[图神经网络]
        SC[科学计算]
    end

    subgraph API["API 层"]
        spmv_csr[spmv_csr]
        spmv_ell[spmv_ell]
        benchmark[benchmark]
        pagerank[pagerank]
    end

    subgraph Kernel["Kernel 层"]
        Scalar["Scalar CSR"]
        Vector["Vector CSR"]
        Merge["Merge Path"]
        ELL["ELL Kernel"]
    end

    subgraph Storage["存储层"]
        CSR_M["CSR Matrix"]
        ELL_M["ELL Matrix"]
    end

    Application --> API
    API --> Kernel
    Kernel --> Storage
```

## 应用场景

- 🕸️ **图算法** — PageRank、最短路径、社区发现
- 🔬 **科学计算** — 有限元分析、计算流体力学
- 🤖 **机器学习** — 稀疏神经网络、推荐系统
- 📊 **数据分析** — 矩阵分解、特征值计算

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
