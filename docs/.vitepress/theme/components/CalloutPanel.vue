<script setup lang="ts">
withDefaults(
  defineProps<{
    title: string
    tone?: 'info' | 'success' | 'warning'
  }>(),
  {
    tone: 'info'
  }
)
</script>

<template>
  <aside class="spmv-callout" :data-tone="tone">
    <div class="spmv-callout-title">{{ title }}</div>
    <div class="spmv-callout-body">
      <slot />
    </div>
  </aside>
</template>
