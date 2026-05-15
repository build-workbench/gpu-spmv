---
layout: home
title: GPU SpMV 技术白皮书
---

<script setup lang="ts">
import { siteData } from '../.vitepress/data/site'
</script>

<HeroEvidence
  eyebrow="Technical Whitepaper"
  :title="siteData.zh.heroTitle"
  :lead="siteData.zh.heroLead"
  :metrics="siteData.zh.metrics"
  primary-label="阅读白皮书"
  primary-link="/zh/whitepaper/"
  secondary-label="查看架构"
  secondary-link="/zh/architecture/overview"
>
  <ThemeAwareArt
    title="SpMV as an Engineering Artifact"
    caption="性能、架构、验证链路与引用体系被一起展示，而不是只给源码。"
  />
</HeroEvidence>

<WhitepaperSection
  eyebrow="Architecture"
  title="先给结论，再给证据，再给设计"
  lead="首页的任务不是罗列细节，而是帮助读者快速判断：这个项目值不值得深入读。"
>
  <ArchitectureCanvas variant="overview-zh" />
</WhitepaperSection>

<WhitepaperSection
  eyebrow="Highlights"
  title="为什么这个项目适合拿来展示"
  lead="因为它不仅讲 CUDA kernel，还把工程规范、性能推理和文档表达放在了一起。"
>
  <div class="spmv-card-grid cols-3">
    <article class="spmv-surface-card spmv-section">
      <h3>性能导向</h3>
      <p>围绕内存带宽利用率、矩阵分布与 kernel 选择给出明确论证。</p>
    </article>
    <article class="spmv-surface-card spmv-section">
      <h3>工程可解释</h3>
      <p>把执行流水线、数据布局、错误处理与 spec-driven workflow 全部显式化。</p>
    </article>
    <article class="spmv-surface-card spmv-section">
      <h3>适合面试与开源展示</h3>
      <p>首页就能看到项目定位、亮点、证据链与延伸阅读路径。</p>
    </article>
  </div>
</WhitepaperSection>
