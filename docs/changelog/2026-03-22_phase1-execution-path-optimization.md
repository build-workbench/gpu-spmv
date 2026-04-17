# Phase 1: 执行路径优化

**日期**: 2026-03-22  
**关联 Issue**: #1, #2  
**里程碑**: v1.1.0 预备

---

## ✨ 新增 (Added)

### SpMV 执行上下文 (`SpMVExecutionContext`)
- 新增 `SpMVExecutionContext` 类，用于纹理对象复用
- 支持 `spmv_csr()` / `spmv_ell()` 在连续调用中复用 texture object
- 避免 benchmark / PageRank 迭代场景中重复创建/销毁纹理对象的开销

### 测试覆盖
- 在 `tests/test_spmv.cu` 中新增执行上下文复用测试
- 在 `tests/test_pagerank.cu` 中新增 dangling nodes 归一化测试

## 🔄 变更 (Changed)

### 源码修改
- `include/spmv/spmv.h`: 增加 `SpMVExecutionContext` 类定义
- `src/spmv_kernels.cu`: 实现 texture context 准备逻辑
- `src/pagerank.cu`: 
  - Dangling sum 累积迁移到 GPU Kernel
  - PageRank update 与 L2 residual 计算迁移到 GPU
  - 仅回传标量结果，消除每轮 Host-Device 整向量往返
- `src/benchmark.cu`: CSR / ELL benchmark 默认复用 `SpMVExecutionContext`

## 🚀 性能

### 优化效果

| 场景 | 优化前 | 优化后 | 提升 |
|:-----|:-------|:-------|:-----|
| PageRank 迭代 | Host-Device 往返 | GPU 全量计算 | ~15-25% |
| 纹理对象创建 | 每次调用创建 | 首次创建后复用 | ~100μs/call |
| Benchmark | 包含开销 | 更接近真实路径 | 更准确 |

---

## 背景

本轮属于深度优化升级的 Phase 1，优先处理当前最明确的固定开销：

1. **SpMV 纹理对象重复创建**: ~100μs/次的开销，在迭代算法中累积明显
2. **PageRank 每轮 Host-Device 往返**: 将完整 rank 向量搬回 Host 后再拷回 Device

移除这些固定损耗，为后续 Merge Path 和 benchmark 口径优化提供更稳定的性能基线。

---

## 影响

### 对用户的影响
- ✅ 无需修改现有代码，API 完全向后兼容
- ✅ 新代码可使用 `SpMVExecutionContext` 获得性能提升
- ✅ PageRank 默认获得性能提升

### API 调整
```cpp
// 现有代码继续工作
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);

// 新：可选上下文复用
SpMVExecutionContext ctx;
for (int i = 0; i < 100; i++) {
    result = spmv_csr(csr, d_x, d_y, &config, n, &ctx);  // 更高效
}
```

---

## 测试验证

- [x] SpMV 上下文复用测试通过
- [x] PageRank dangling nodes 测试通过
- [x] 所有现有测试保持通过
- [x] Benchmark 结果符合预期
