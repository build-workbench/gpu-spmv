# 引用格式

如果您在研究中使用 GPU SpMV，请引用：

## BibTeX 格式

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

## 文本格式

```
LessUp. GPU SpMV: High-Performance CUDA Sparse Matrix-Vector Multiplication.
GitHub repository, 2026. https://github.com/LessUp/gpu-spmv
```

---

## 相关论文

本库实现的算法基于以下研究：

### Merge Path 算法

1. **Merrill, D., & Garland, M. (2016)**. Merge-based parallel sparse matrix-vector multiplication. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '16)*. IEEE.

   ::: tip 核心贡献
   Merge Path 算法通过基于行指针和工作索引的合并操作进行工作分区，为不规则稀疏矩阵实现完美负载均衡。
   :::

### 向量化 CSR

2. **Bell, N., & Garland, M. (2009)**. Implementing sparse matrix-vector multiplication on throughput-oriented processors. *Proceedings of SC '09*. IEEE.

3. **Bell, N., Dalton, S., & Olson, L. N. (2012)**. Exposing fine-grained parallelism in algebraic multigrid methods. *SIAM Journal on Scientific Computing*, 34(4), C170-C194.

### ELL 格式

4. **Vázquez, F., Fernández, J. J., & Garzón, E. M. (2011)**. Automatic tuning of the sparse matrix vector product on GPUs based on the ELL-R-T format. *Concurrency and Computation: Practice and Experience*, 24(1), 1-20.

---

## 算法参考

| 算法 | 参考文献 | 核心思想 |
|:-----|:---------|:---------|
| Scalar CSR | Bell & Garland (2009) | 每行一线程 |
| Vector CSR | Bell & Garland (2009) | 每行一 warp |
| Merge Path | Merrill & Garland (2016) | 基于合并的分区 |
| ELL Kernel | Vázquez et al. (2011) | 列主序合并访存 |

---

## 基准测试方法

我们的基准测试方法遵循以下最佳实践：

- **SPAPT 基准测试套件**：稀疏计算标准化性能评估
- **SuiteSparse 矩阵集**：真实世界测试矩阵
- **GPU 性能指标**：内存带宽利用率作为主要指标

---

## 致谢

本库基于 CUDA 生态系统的优秀工作：

- NVIDIA cuSPARSE 提供参考实现
- Thrust 库提供并行原语
- Google Test 提供测试基础设施