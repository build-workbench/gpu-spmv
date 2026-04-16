---
layout: default
title: Architecture
parent: Documentation
nav_order: 2
---

# 🏗️ Architecture
{: .no_toc }

System architecture, core algorithms, and design decisions.
{: .fs-6 .fw-300 }

## Table of Contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## System Overview

```
┌────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├────────────────────────────────────────────────────────────┤
│                    SpMV Runtime API                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ spmv_csr │ │ spmv_ell │ │ benchmark│ │   pagerank   │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
├────────────────────────────────────────────────────────────┤
│                    Kernel Scheduler                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │  Scalar  │ │  Vector  │ │  Merge   │ │     ELL      │    │
│  │   CSR    │ │   CSR    │ │   Path   │ │   Kernel     │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
├────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  ┌────────────────┐    ┌────────────────┐                   │
│  │  CSR Matrix    │    │  ELL Matrix    │                   │
│  └────────────────┘    └────────────────┘                   │
└────────────────────────────────────────────────────────────┘
```

---

## Sparse Matrix Formats

### CSR

Uses three arrays: `values`, `col_indices`, `row_ptrs`.

### ELL

Column-major storage for coalesced GPU memory access.

---

## Kernel Selection

| Condition | Kernel | Description |
|:----------|:-------|:------------|
| `avg_nnz < 4` | Scalar CSR | Single thread per row |
| `skewness < 10` | Vector CSR | Warp collaborative |
| `skewness >= 10` | Merge Path | Perfect load balancing |
| ELL format | ELL | Coalesced access |

---

## Design Decisions

### RAII Resource Management

Automatic lifecycle management prevents memory leaks.

### Execution Context Reuse

Texture objects are expensive to create. Reuse them across calls.
