---
layout: home
hero:
  name: GPU SpMV
  text: ' '
  actions:
    - theme: brand
      text: 简体中文
      link: /zh/
    - theme: alt
      text: English
      link: /en/
---

<script setup>
import { onMounted } from 'vue'
import { useRouter, useData } from 'vitepress'

onMounted(() => {
  const router = useRouter()
  const { site } = useData()
  const base = site.value.base
  const lang = navigator.language || navigator.userLanguage
  if (lang.startsWith('zh')) {
    router.go(`${base}zh/`)
  } else {
    router.go(`${base}en/`)
  }
})
</script>
