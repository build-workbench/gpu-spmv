<script setup lang="ts">
defineProps<{
  items: Array<{
    label: string
    value: string
    description?: string
  }>
}>()
</script>

<template>
  <div class="spmv-metric-strip">
    <article v-for="item in items" :key="item.label" class="spmv-metric-card spmv-surface-card">
      <div class="spmv-metric-value">{{ item.value }}</div>
      <div class="spmv-metric-label">{{ item.label }}</div>
      <p v-if="item.description">{{ item.description }}</p>
    </article>
  </div>
</template>
