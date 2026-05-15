<script setup lang="ts">
defineProps<{
  title: string
  caption: string
}>()
</script>

<template>
  <svg viewBox="0 0 320 180" class="theme-aware-art" role="img" :aria-label="title">
    <rect x="0" y="0" width="320" height="180" rx="24" class="art-bg" />
    <rect x="24" y="24" width="48" height="48" rx="8" class="art-accent-1" />
    <rect x="88" y="40" width="24" height="24" rx="6" class="art-accent-2" />
    <rect x="128" y="24" width="72" height="72" rx="16" class="art-accent-1" opacity="0.25" />
    <text x="24" y="130" class="art-title">{{ title }}</text>
    <text x="24" y="154" class="art-caption">{{ caption }}</text>
  </svg>
</template>
