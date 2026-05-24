# Execution Pipeline

## Why this deserves its own page

GPU SpMV is not just “launch a kernel.” The real engineering story is **how the matrix is analyzed, how kernel choice is made, how execution context is reused, and how the result is interpreted with confidence**.

## Pipeline Breakdown

1. **Input stage**: load CSR / ELL data structures and prepare the input vector.
2. **Analysis stage**: compute `avg_nnz_per_row`, skewness, and row distribution characteristics.
3. **Decision stage**: choose Scalar CSR, Vector CSR, Merge Path, or ELL.
4. **Execution stage**: launch the GPU kernel and record timing / bandwidth metrics.
5. **Validation stage**: compare against CPU reference behavior or established baselines.

## Key Decisions

| Observation | Decision |
|:------------|:---------|
| `avg_nnz_per_row < 4` | Scalar CSR to avoid wasting warp-scale resources |
| Rows are uniform and low-skew | Vector CSR for stronger warp collaboration |
| Row lengths are highly skewed | Merge Path to prioritize load balance |
| Row width is nearly fixed | ELL kernel to prioritize coalesced access |

## Read this together with

- [Kernel Selection](/en/architecture/kernel-selection)
- [Memory Layout](/en/architecture/memory-layout)
- [Performance Methodology](/en/performance/methodology)
