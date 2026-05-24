<script setup lang="ts">
defineProps<{
  items: Array<{
    key: string
    title: string
    meta?: string
    url: string
  }>
}>()
</script>

<template>
  <div class="spmv-citation-grid">
    <article v-for="item in items" :key="item.key" class="spmv-citation-card spmv-surface-card">
      <h3>{{ item.title }}</h3>
      <p v-if="item.meta">{{ item.meta }}</p>
      <a :href="item.url" target="_blank" rel="noreferrer">Open reference</a>
    </article>
  </div>
</template>
