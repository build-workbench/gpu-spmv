# 学术参考

GPU SpMV 的实现基于以下学术研究成果。

## 核心算法

### Merge-based Parallel SpMV

> Merrill, D., & Garland, M. (2016). **Merge-based parallel sparse matrix-vector multiplication**. *ACM SIGPLAN Notices*, 51(8), 12-21.

- **贡献**: 提出 Merge Path 算法，实现完美负载均衡
- **应用于**: `MERGE_PATH` kernel
- [DOI: 10.1145/3016078.285114](https://doi.org/10.1145/3016078.285114)

### Bell & Garland Survey

> Bell, N., & Garland, M. (2009). **Implementing sparse matrix-vector multiplication on throughput-oriented processors**. *SC'09: Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis*.

- **贡献**: CSR vs ELL 格式性能分析，GPU SpMV 基础理论
- **应用于**: `VECTOR_CSR`、`ELL_KERNEL` 设计
- [DOI: 10.1145/1654059.1654121](https://doi.org/10.1145/1654059.1654121)

### CSR5 Format

> Liu, Y., & Vuduc, R. (2018). **An adaptive algorithm for sparse matrix-vector multiplication on GPUs**. *IEEE Transactions on Parallel and Distributed Systems*.

- **贡献**: CSR5 格式，自适应负载均衡
- **参考**: 理解不规则稀疏矩阵的负载分布

## GPU 计算基础

### CUDA Best Practices

> NVIDIA. (2024). **CUDA C++ Best Practices Guide**.

- **参考**: 内存合并、纹理缓存、Warp 同步
- [Link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### CUDA Programming Guide

> NVIDIA. (2024). **CUDA C++ Programming Guide**.

- **参考**: CUDA 执行模型、存储层次
- [Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## PageRank

> Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). **The PageRank citation ranking: Bringing order to the web**. *Stanford InfoLab*.

- **贡献**: PageRank 算法原始论文
- **应用于**: `pagerank()` 实现

## 相关项目

| Project | Stars | Description | Key Takeaway |
|:--------|:-----:|:------------|:-------------|
| [Ginkgo](https://github.com/ginkgo-project/ginkgo) | 597 | High-performance linear algebra | Performance visualization |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | N/A | NVIDIA official library | Performance baseline |
| [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) | 947 | Sparse matrix collection | Standard test data |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) | 300+ | Multi-backend sparse BLAS | Performance portability |

## 引用本项目

```bibtex
@software{gpuspmv2024,
  author = {LessUp},
  title = {GPU SpMV: High-Performance CUDA Sparse Matrix-Vector Multiplication},
  year = {2024},
  url = {https://github.com/LessUp/gpu-spmv}
}
```

## 推荐阅读

1. **GPU Architecture**: 了解 GPU 内存层次和执行模型
2. **Sparse Matrix Formats**: 不同格式的优缺点
3. **Load Balancing**: 并行计算中的负载均衡技术
4. **Memory Coalescing**: GPU 内存访问优化
