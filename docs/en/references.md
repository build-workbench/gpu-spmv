# References

<script setup lang="ts">
import { references } from '../.vitepress/data/references'
</script>

This page separates papers, comparable projects, and follow-up reading so readers can understand **what this project learned from and what ecosystem it belongs to**.

## Core Papers

<CitationGrid :items="references.papers" />

## Representative Projects

<CitationGrid :items="references.projects" />

## How to read these references

1. Start with **Bell & Garland** for the classic GPU SpMV framing.
2. Read **Merrill & Garland** to understand why Merge Path matters for irregular work distribution.
3. Compare against **cuSPARSE / Ginkgo / SuiteSparse** to place this project inside the real sparse-computing ecosystem.
