---
layout: default
title: 性能优化
---

# 性能优化

## Kernel 选择策略

SpMV 的性能高度依赖矩阵的非零元分布。本库提供自动 Kernel 选择机制，根据矩阵统计特征选取最优实现。

### 决策逻辑

```
矩阵特征分析
    │
    ├── avg_nnz_per_row < 4  ──→  Scalar CSR
    │       每行非零元少，线程级并行即可
    │
    ├── skewness < 10        ──→  Vector CSR
    │       行长度较均匀，Warp 级并行高效
    │
    └── skewness >= 10       ──→  Merge Path
            行长度差异大，均匀切分工作量
```

### Scalar CSR

- **策略**：一个线程处理一行
- **适用**：稀疏度极高、每行非零元很少的矩阵
- **优势**：线程开销最小，无 Warp 内同步

### Vector CSR

- **策略**：一个 Warp（32 线程）协作处理一行
- **适用**：行长度中等且分布均匀
- **优势**：Warp 内归约高效，合并内存访问

### Merge Path

- **策略**：将行指针和非零元序列视为两条路径，二维切分保证每个线程块处理相同数量的工作
- **适用**：行长度分布极不均匀（如幂律图）
- **优势**：完美负载均衡，避免长行成为瓶颈

### ELL Kernel

- **策略**：Column-major 存储，一个线程处理一行
- **适用**：所有行长度接近的矩阵
- **优势**：访存完全合并，带宽利用率最高

## 带宽优化

### Column-Major 存储（ELL）

ELL 格式使用 column-major 布局，使得相邻线程访问连续内存地址，实现完全合并的全局内存访问：

```
Row-major (差):  T0→[r0c0] T1→[r1c0]  不连续
Col-major (好):  T0→[r0c0] T1→[r1c0]  连续地址
```

### 纹理缓存

对输入向量 `x` 的访问通常是随机模式（由列索引决定）。启用 `use_texture = true` 后，通过纹理缓存读取 `x`，利用空间局部性缓存提升命中率。

### Bank Conflict 消除

Shared Memory 的 bank 冲突会严重影响归约操作性能。Vector CSR 的 Warp 归约使用 shuffle 指令，完全避免 Shared Memory bank 冲突。

## 基准测试

### 使用方法

```bash
# 构建后运行
./build/spmv_benchmark
```

### 度量指标

| 指标 | 说明 |
|------|------|
| `avg_time_ms` | 平均执行时间（不含 warmup） |
| `gflops` | 每秒十亿浮点运算 |
| `bandwidth_gb_s` | 实际内存带宽（GB/s） |
| `bandwidth_utilization` | 带宽利用率 = 实际 / 峰值 |

### 峰值带宽缓存

`get_gpu_peak_bandwidth()` 使用 `std::call_once` 缓存查询结果，避免重复的 CUDA API 调用。

### CPU 计时

基准测试的外层计时使用 `std::chrono::high_resolution_clock`，而非 CUDA Events，确保包含所有 host-side 开销。

## 性能调优建议

1. **选择合适格式** — 行长度均匀用 ELL，不均匀用 CSR
2. **启用纹理缓存** — 对大规模随机访问的输入向量有效
3. **复用 CudaBuffer** — 避免重复分配/释放 GPU 内存
4. **使用自动配置** — `spmv_auto_config()` 根据矩阵特征选择最优 Kernel
5. **检查带宽利用率** — 利用率 < 50% 通常意味着有优化空间
