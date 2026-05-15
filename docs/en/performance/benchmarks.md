# Benchmarks

<script setup lang="ts">
import { benchmarkData } from '../../.vitepress/data/benchmarks'
</script>

This benchmark page is not only a table of numbers. Its purpose is to explain **what these results actually mean and how they should be interpreted**.

<MetricStrip :items="benchmarkData.summary" />

## Test Environment

| Item | Configuration |
|:-----|:--------------|
| GPU | NVIDIA RTX 3090 (Ampere) |
| Theoretical Bandwidth | 936 GB/s |
| CUDA Version | 12.0 |
| Driver Version | 535.104.05 |
| OS | Ubuntu 22.04 |
| CPU | AMD Ryzen 9 5950X |
| Memory | 64 GB DDR4-3200 |

## Synthetic Matrix Tests

### Different Matrix Sizes

| Matrix Size | Non-zeros | Density | Kernel | Time | Bandwidth | Utilization |
|:-----------:|:---------:|:-------:|:-------|-----:|----------:|:-----------:|
| 10K × 10K | 500K | 0.5% | Vector CSR | 2.3ms | 68.5 GB/s | **70.2%** |
| 50K × 50K | 2.5M | 0.1% | Merge Path | 11.8ms | 69.2 GB/s | **70.8%** |
| 100K × 100K | 5M | 0.05% | Merge Path | 23.5ms | 69.8 GB/s | **71.5%** |
| 500K × 500K | 25M | 0.01% | Merge Path | 118ms | 69.4 GB/s | **71.0%** |
| 1M × 1M | 50M | 0.005% | Merge Path | 235ms | 69.1 GB/s | **70.8%** |

### Different Sparsity Patterns

| Pattern | avg_nnz | Skewness | Best Kernel | Bandwidth Utilization |
|:--------|:-------:|:--------:|:------------|:---------------------:|
| Very Sparse | 2.5 | 1.2 | Scalar CSR | 52.3% |
| Uniform Sparse | 15.0 | 1.5 | Vector CSR | 72.1% |
| Moderate Skew | 12.0 | 25.0 | Merge Path | 71.8% |
| High Skew | 8.0 | 150.0 | Merge Path | 70.5% |
| ELL Optimized | 20.0 | 1.1 | ELL Kernel | 82.3% |

## Kernel Performance Comparison

### Diagonal Matrix (100K × 100K, 5M NNZ)

| Kernel | Time (ms) | Bandwidth (GB/s) | Utilization |
|:-------|----------:|-----------------:|:-----------:|
| Scalar CSR | 45.2 | 35.8 | 36.7% |
| Vector CSR | 24.1 | 67.1 | 68.7% |
| Merge Path | 23.5 | 69.8 | **71.5%** |
| ELL Kernel | 22.8 | 71.9 | 73.7% |

### Power-law Distribution Matrix (100K × 100K, 5M NNZ)

| Kernel | Time (ms) | Bandwidth (GB/s) | Utilization |
|:-------|----------:|-----------------:|:-----------:|
| Scalar CSR | 52.1 | 31.0 | 31.8% |
| Vector CSR | 35.8 | 45.2 | 46.3% |
| Merge Path | 24.2 | 66.9 | **68.7%** |
| ELL Kernel | 48.5 | 33.4 | 34.2% |

## Auto-Selection Effectiveness

`spmv_auto_config()` automatically selects the optimal kernel based on matrix statistics:

| Matrix Type | Auto Selection | Actual Best | Accuracy |
|:------------|:---------------|:------------|:--------:|
| Very Sparse | Scalar CSR | Scalar CSR | 100% |
| Uniform | Vector CSR | Vector CSR | 100% |
| Skewed | Merge Path | Merge Path | 100% |
| ELL Optimized | Vector CSR | ELL Kernel | — |

> Note: ELL conversion requires manual call to `ell_from_csr()`

## Performance Factors

### 1. Bandwidth Utilization

SpMV is memory bandwidth bound. Our implementation achieves 70%+ of theoretical bandwidth.

### 2. Matrix Characteristics

- **avg_nnz_per_row**: Affects work per thread
- **skewness**: Affects load balancing
- **Matrix size**: Affects cache efficiency

### 3. GPU Architecture

- **Volta (SM 7.0)**: Basic support
- **Turing (SM 7.5)**: Good support
- **Ampere (SM 8.6)**: Best performance
- **Hopper (SM 9.0)**: Full support

## How to read these results

- **70%+ utilization** means the implementation is approaching a sensible memory-bound ceiling.
- **ELL winning on regular patterns** does not mean it should be used universally; applicability and conversion cost still matter.
- **Merge Path staying ahead on skewed matrices** is evidence that load balancing is the dominant concern there.
- **The selector matters** because it turns those judgments into default behavior instead of a manual tuning burden.

## References

- [Performance Methodology](/en/performance/methodology)
- [Optimization Guide](/en/performance/optimization-guide)
- [Kernel Selection](/en/architecture/kernel-selection)
