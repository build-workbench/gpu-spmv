# Citation

If you use GPU SpMV in your research, please cite:

## BibTeX

```bibtex
@software{gpu_spmv_2026,
  author = {LessUp},
  title = {GPU SpMV: High-Performance CUDA Sparse Matrix-Vector Multiplication},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/LessUp/gpu-spmv},
  version = {1.0.0}
}
```

## Text Format

```
LessUp. GPU SpMV: High-Performance CUDA Sparse Matrix-Vector Multiplication.
GitHub repository, 2026. https://github.com/LessUp/gpu-spmv
```

---

## Related Publications

The algorithms implemented in this library are based on the following research:

### Merge Path Algorithm

1. **Merrill, D., & Garland, M. (2016)**. Merge-based parallel sparse matrix-vector multiplication. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16)*. IEEE.

   ::: tip Key Contribution
   The Merge Path algorithm enables perfect load balancing for irregular sparse matrices by partitioning work based on the merge operation between row pointers and work indices.
   :::

### Vectorized CSR

2. **Bell, N., & Garland, M. (2009)**. Implementing sparse matrix-vector multiplication on throughput-oriented processors. *Proceedings of SC '09*. IEEE.

3. **Bell, N., Dalton, S., & Olson, L. N. (2012)**. Exposing fine-grained parallelism in algebraic multigrid methods. *SIAM Journal on Scientific Computing*, 34(4), C170-C194.

### ELL Format

4. **Vázquez, F., Fernández, J. J., & Garzón, E. M. (2011)**. Automatic tuning of the sparse matrix vector product on GPUs based on the ELL-R-T format. *Concurrency and Computation: Practice and Experience*, 24(1), 1-20.

---

## Algorithm References

| Algorithm | Reference | Key Idea |
|:----------|:----------|:---------|
| Scalar CSR | Bell & Garland (2009) | One thread per row |
| Vector CSR | Bell & Garland (2009) | One warp per row |
| Merge Path | Merrill & Garland (2016) | Merge-based partitioning |
| ELL Kernel | Vázquez et al. (2011) | Column-major coalesced access |

---

## Benchmark Methodology

Our benchmark methodology follows best practices from:

- **SPAPT Benchmark Suite**: Standardized performance assessment for sparse computations
- **SuiteSparse Matrix Collection**: Real-world test matrices
- **GPU Performance Metrics**: Memory bandwidth utilization as primary metric

---

## Acknowledgments

This library builds upon the excellent work of the CUDA ecosystem:

- NVIDIA cuSPARSE for reference implementations
- Thrust library for parallel primitives
- Google Test for testing infrastructure