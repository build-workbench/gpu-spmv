# 学术参考

<script setup lang="ts">
import { references } from '../.vitepress/data/references'
</script>

本页把论文、项目和延伸阅读分开整理，方便读者快速建立“**这个项目参考了谁、站在什么技术谱系上**”的理解。

## 核心论文

<CitationGrid :items="references.papers" />

## 代表性项目

<CitationGrid :items="references.projects" />

## 如何使用这些参考

1. **先读 Bell & Garland**，理解 GPU SpMV 的经典问题定义。
2. **再看 Merrill & Garland**，理解 Merge Path 在不规则负载中的价值。
3. **对照 cuSPARSE / Ginkgo / SuiteSparse**，把本项目放回真实工程生态里看。
