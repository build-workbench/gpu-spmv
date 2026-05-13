# Academic References

GPU SpMV is built upon the following academic research.

## Core Algorithms

### Merge-based Parallel SpMV

> Merrill, D., & Garland, M. (2016). **Merge-based parallel sparse matrix-vector multiplication**. *ACM SIGPLAN Notices*, 51(8), 12-21.

- **Contribution**: Proposed the Merge Path algorithm for perfect load balancing
- **Applied to**: `MERGE_PATH` kernel
- [DOI: 10.1145/3016078.285114](https://doi.org/10.1145/3016078.285114)

### Bell & Garland Survey

> Bell, N., & Garland, M. (2009). **Implementing sparse matrix-vector multiplication on throughput-oriented processors**. *SC'09: Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis*.

- **Contribution**: CSR vs ELL format performance analysis, foundational GPU SpMV theory
- **Applied to**: `VECTOR_CSR`, `ELL_KERNEL` design
- [DOI: 10.1145/1654059.1654121](https://doi.org/10.1145/1654059.1654121)

### CSR5 Format

> Liu, Y., & Vuduc, R. (2018). **An adaptive algorithm for sparse matrix-vector multiplication on GPUs**. *IEEE Transactions on Parallel and Distributed Systems*.

- **Contribution**: CSR5 format with adaptive load balancing
- **Reference**: Understanding load distribution in irregular sparse matrices

## GPU Computing

### CUDA Best Practices

> NVIDIA. (2024). **CUDA C++ Best Practices Guide**.

- **Reference**: Memory coalescing, texture cache, warp synchronization
- [Link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### CUDA Programming Guide

> NVIDIA. (2024). **CUDA C++ Programming Guide**.

- **Reference**: CUDA execution model, memory hierarchy
- [Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## PageRank

> Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). **The PageRank citation ranking: Bringing order to the web**. *Stanford InfoLab*.

- **Contribution**: Original PageRank algorithm
- **Applied to**: `pagerank()` implementation

## Related Projects

| Project | Stars | Description | Key Takeaway |
|:--------|:-----:|:------------|:-------------|
| [Ginkgo](https://github.com/ginkgo-project/ginkgo) | 597 | High-performance linear algebra | Performance visualization |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | N/A | NVIDIA official library | Performance baseline |
| [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) | 947 | Sparse matrix collection | Standard test data |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) | 300+ | Multi-backend sparse BLAS | Performance portability |

## Cite This Project

```bibtex
@software{gpuspmv2024,
  author = {LessUp},
  title = {GPU SpMV: High-Performance CUDA Sparse Matrix-Vector Multiplication},
  year = {2024},
  url = {https://github.com/LessUp/gpu-spmv}
}
```

## Further Reading

1. **GPU Architecture**: Understanding GPU memory hierarchy and execution model
2. **Sparse Matrix Formats**: Trade-offs between different formats
3. **Load Balancing**: Techniques for parallel load balancing
4. **Memory Coalescing**: GPU memory access optimization
