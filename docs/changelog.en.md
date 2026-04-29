---
layout: default
title: Changelog
nav_order: 7
permalink: /changelog.en
lang: en
---

<p align="right">
  <a href="changelog">🇨🇳 简体中文</a>
</p>

# Changelog
{: .no_toc }

Version history and updates for GPU SpMV.
{: .fs-6 .fw-300 }

---

## v1.1.0 (2026-04-22)

### Added

- 🎨 **New GitHub Pages Site** - Redesigned documentation website
- 🚀 **Performance Visualization** - Added performance comparison charts
- 📚 **Bilingual Docs** - Complete Chinese and English documentation support

### Improved

- Redesigned homepage highlighting product features
- Unified content depth across Chinese and English versions
- Optimized navigation structure

---

## v1.0.1 (2026-03-22)

### Improved

- **Kernel Selection** - Improved accuracy of auto-selection algorithm
- **Memory Management** - Optimized CudaBuffer move semantics
- **Build Speed** - Improved CMake configuration for faster builds

### Fixed

- Fixed boundary check issue in ELL format conversion
- Fixed numerical stability in PageRank under specific convergence conditions

---

## v1.0.0 (2026-03-10)

### First Release

- 🎉 **First Stable Release**
- 4 optimized kernels (Scalar/Vector/Merge Path/ELL)
- Dual format support (CSR and ELL)
- Complete RAII resource management
- 100+ test cases coverage
- PageRank algorithm implementation

### Features

- Intelligent automatic kernel selection
- Up to 70%+ bandwidth utilization
- Cross-platform support (Linux/Windows)
- CMake Presets one-click build

---

<div class="text-center text-small" style="margin-top: 3rem;">
  <p>Full release notes on <a href="https://github.com/LessUp/gpu-spmv/releases">GitHub Releases</a></p>
</div>
