# Add Automatic Kernel Selection

## Why

需要根据矩阵特征自动选择最优的 SpMV 内核，避免用户手动选择的复杂性，并确保在不同矩阵模式下都能获得最佳性能。

## What Changes

### New Capabilities
- `kernel-selection` - 自动内核选择策略

### Modified Capabilities
- `spmv-kernels` - 添加自动选择函数

## Impact

**New Files:**
- `tests/test_kernel_selector.cpp` - 选择器测试

**Modified Files:**
- `include/spmv/spmv.h` - 添加 `spmv_auto_config()`
- `src/spmv_kernels.cu` - 添加矩阵统计计算

**Selection Heuristic:**
```
avg_nnz_per_row < 4 → SCALAR_CSR
avg_nnz_per_row >= 4 AND skewness < 10 → VECTOR_CSR
avg_nnz_per_row >= 4 AND skewness >= 10 → MERGE_PATH
```

## Status

✅ Completed - 2025-03-01
