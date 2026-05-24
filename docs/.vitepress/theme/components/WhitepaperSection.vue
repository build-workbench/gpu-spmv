<script setup lang="ts">
withDefaults(
  defineProps<{
    title: string
    eyebrow?: string
    lead?: string
  }>(),
  {
    eyebrow: 'Section',
    lead: ''
  }
)
</script>

<template>
  <section class="spmv-section spmv-surface-card">
    <div class="spmv-eyebrow">{{ eyebrow }}</div>
    <h2>{{ title }}</h2>
    <p v-if="lead">{{ lead }}</p>
    <div class="spmv-section-body">
      <slot />
    </div>
  </section>
</template>
