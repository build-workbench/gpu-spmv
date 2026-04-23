# Add PageRank Algorithm

## Why

需要实现 PageRank 算法以展示稀疏矩阵操作在图数据上的实际应用。PageRank 是一个经典的迭代 SpMV 应用，广泛用于网页排名、社交网络分析等领域。

## What Changes

### New Capabilities
- `pagerank` - PageRank 图算法实现

### Modified Capabilities
- `spmv-kernels` - 使用 SpMV 作为核心操作

## Impact

**New Files:**
- `include/spmv/pagerank.h` - PageRank 接口头文件
- `src/pagerank.cu` - PageRank 实现
- `tests/test_pagerank.cu` - PageRank 测试

**Features:**
- 迭代式 PageRank 计算
- 阻尼因子配置 (默认 0.85)
- 收敛检测 (L2 范数 < 1e-6)
- 悬挂节点处理
- Top-K 节点输出
- 支持百万级节点图

**Algorithm:**
```
r_{k+1} = d × A × r_k + (1-d) / n
```

## Status

✅ Completed - 2025-03-10
