<script setup lang="ts">
type MetricItem = {
  label: string
  value: string
}

withDefaults(
  defineProps<{
    title: string
    lead: string
    metrics: MetricItem[]
    eyebrow?: string
    primaryLabel: string
    primaryLink: string
    secondaryLabel?: string
    secondaryLink?: string
  }>(),
  {
    eyebrow: 'Technical Whitepaper'
  }
)
</script>

<template>
  <section class="spmv-hero">
    <div class="spmv-hero-copy spmv-surface-card">
      <div class="spmv-eyebrow">{{ eyebrow }}</div>
      <h1>{{ title }}</h1>
      <p>{{ lead }}</p>
      <div class="spmv-hero-actions">
        <a class="primary" :href="primaryLink">{{ primaryLabel }}</a>
        <a v-if="secondaryLabel && secondaryLink" class="secondary" :href="secondaryLink">
          {{ secondaryLabel }}
        </a>
      </div>
      <div class="spmv-metric-strip">
        <article v-for="metric in metrics" :key="metric.label" class="spmv-metric-card spmv-surface-card">
          <div class="spmv-metric-value">{{ metric.value }}</div>
          <div class="spmv-metric-label">{{ metric.label }}</div>
        </article>
      </div>
    </div>
    <div class="spmv-hero-art spmv-surface-card">
      <slot />
    </div>
  </section>
</template>
