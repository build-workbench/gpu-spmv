# 优化指南

如何获得 GPU SpMV 的最佳性能。

## 1. 选择正确的 Kernel

### 自动选择

```cpp
SpMVConfig config = spmv_auto_config(csr);  // 推荐
```

### 手动选择

| 场景 | 推荐 Kernel | 原因 |
|:-----|:------------|:-----|
| 极稀疏 (avg_nnz < 4) | Scalar CSR | 最小开销 |
| 均匀分布 | Vector CSR | Warp 协作效率高 |
| 高度倾斜 | Merge Path | 完美负载均衡 |
| 行长度均匀 | ELL Kernel | 完全合并访存 |

## 2. 使用 ELL 格式

当矩阵行长度均匀时，ELL 格式可获得最佳性能：

```cpp
CSRStats stats = csr_compute_stats(csr);

if (stats.skewness < 3.0f) {
    // 转换为 ELL
    ELLMatrix* ell = ell_create(csr->num_rows, csr->num_cols,
                                stats.max_nnz_per_row);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);

    // 使用 ELL SpMV
    result = spmv_ell(ell, d_x, d_y, nullptr);
}
```

## 3. 复用执行上下文

迭代计算中复用纹理缓存：

```cpp
SpMVExecutionContext ctx;  // 创建一次

for (int i = 0; i < iterations; i++) {
    // 纹理对象被复用，避免重复创建
    result = spmv_csr(csr, d_x, d_y, &config, n, &ctx);
}

ctx.reset();  // 完成后重置
```

## 4. 内存布局优化

### 向量对齐

```cpp
// 使用 CudaBuffer 确保对齐
CudaBuffer<float> d_x(N);
CudaBuffer<float> d_y(N);
```

### 批量传输

```cpp
// 批量传输多个向量
cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
// 而不是多次小传输
```

## 5. 调整阈值

根据特定硬件调整选择阈值：

```cpp
SpMVThresholds thresholds = spmv_get_thresholds();

// 在较新 GPU 上可能需要调整
thresholds.avg_nnz_threshold = 3.0f;      // 降低 Scalar CSR 使用
thresholds.skewness_threshold = 15.0f;    // 提高 Merge Path 使用

spmv_set_thresholds(thresholds);
```

## 6. 性能分析

### 自建简单计时循环

```cpp
SpMVExecutionContext ctx;
SpMVConfig config = spmv_auto_config(csr);

for (int i = 0; i < 5; ++i) {
    spmv_csr(csr, d_x, d_y, &config, csr->num_cols, &ctx);  // 预热
}

SpMVResult result = spmv_csr(csr, d_x, d_y, &config, csr->num_cols, &ctx);
printf("Elapsed: %.3f ms\n", result.elapsed_ms);
printf("Bandwidth: %.1f GB/s\n", result.bandwidth_gb_s);
```

### 使用 Nsight

```bash
# 性能分析
nsys profile ./spmv_program

# 详细分析
ncu ./spmv_program
```

## 7. 常见问题

### 带宽利用率低

**原因**: Kernel 选择不当或内存访问模式差

**解决**:
1. 检查矩阵统计量
2. 尝试手动指定 Kernel
3. 考虑转换为 ELL 格式

### 性能波动大

**原因**: GPU 频率波动或内存碎片

**解决**:
1. 固定 GPU 频率：`sudo nvidia-smi -lgc 1800`
2. 预热运行：执行几次后再计时
3. 使用多次运行取最小值

### ELL 转换后性能下降

**原因**: 矩阵行长度差异大，填充过多

**解决**:
1. 检查 skewness，仅当 < 3 时转换
2. 考虑 HYB 格式（ELL + COO 混合）

## 性能检查清单

- [ ] 使用 `spmv_auto_config()` 自动选择
- [ ] 检查是否适合 ELL 格式
- [ ] 迭代计算中复用执行上下文
- [ ] 使用 `CudaBuffer` 管理内存
- [ ] 验证带宽利用率 > 60%

## 参考

- [基准测试](/zh/performance/benchmarks)
- [Kernel 选择策略](/zh/architecture/kernel-selection)