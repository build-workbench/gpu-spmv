---
layout: home
title: GPU SpMV Technical Whitepaper
---

<script setup lang="ts">
import { siteData } from '../.vitepress/data/site'
</script>

<HeroEvidence
  eyebrow="Technical Whitepaper"
  :title="siteData.en.heroTitle"
  :lead="siteData.en.heroLead"
  :metrics="siteData.en.metrics"
  primary-label="Read the Whitepaper"
  primary-link="/en/whitepaper/"
  secondary-label="View Architecture"
  secondary-link="/en/architecture/overview"
>
  <ThemeAwareArt
    title="Readable, Verifiable, Presentable"
    caption="The site explains not only what the project does, but why its design and validation deserve attention."
  />
</HeroEvidence>

<WhitepaperSection
  eyebrow="Architecture"
  title="Lead with conclusions, then evidence, then implementation"
  lead="The landing page should help a reader decide quickly whether this project is worth deeper reading."
>
  <ArchitectureCanvas variant="overview-en" />
</WhitepaperSection>

<WhitepaperSection
  eyebrow="Highlights"
  title="Why this project is strong as a showcase"
  lead="Because it combines CUDA performance work with engineering discipline, explainability, and documentation quality."
>
  <div class="spmv-card-grid cols-3">
    <article class="spmv-surface-card spmv-section">
      <h3>Performance-first</h3>
      <p>Kernel choice, irregular sparsity behavior, and bandwidth utilization are presented as explicit decisions.</p>
    </article>
    <article class="spmv-surface-card spmv-section">
      <h3>Engineering clarity</h3>
      <p>The execution pipeline, memory layout, and reliability story are visible without extra process machinery.</p>
    </article>
    <article class="spmv-surface-card spmv-section">
      <h3>Interview-ready narrative</h3>
      <p>A reviewer can understand the value proposition, evidence chain, and reading path directly from the site.</p>
    </article>
  </div>
</WhitepaperSection>
