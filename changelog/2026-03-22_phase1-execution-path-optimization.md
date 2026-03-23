# Phase 1 执行路径优化

日期：2026-03-22

## 变更内容

- 为 `include/spmv/spmv.h` 增加可复用的 `SpMVExecutionContext`，并让 `spmv_csr()` / `spmv_ell()` 支持在连续调用中复用 texture object
- 在 `src/spmv_kernels.cu` 中实现 texture context 准备逻辑，避免 benchmark / PageRank 这类迭代场景重复创建与销毁纹理对象
- 在 `src/pagerank.cu` 中把 dangling sum 累积、PageRank update 与 L2 residual 计算迁移到 GPU kernel 路径，仅回传标量结果，移除每轮整向量 Host-Device 往返
- 在 `src/benchmark.cu` 中让 CSR / ELL benchmark 默认复用 `SpMVExecutionContext`，使基准结果更贴近优化后的真实执行路径
- 在 `tests/test_spmv.cu` 中新增执行上下文复用测试，在 `tests/test_pagerank.cu` 中新增 dangling nodes 归一化测试

## 背景

本轮属于深度优化升级的 Phase 1，优先处理当前最明确的固定开销：SpMV 纹理对象重复创建，以及 PageRank 每轮迭代将完整 rank 向量搬回 Host 后再拷回 Device。先移除这些固定损耗，可以为后续 Merge Path 和 benchmark 口径优化提供更稳定的性能基线。
