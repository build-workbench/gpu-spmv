# Add SpMV CUDA Kernels

## Why

需要实现 GPU 加速的稀疏矩阵-向量乘法 (SpMV) 内核。SpMV 是许多科学计算和图算法的核心操作，GPU 加速可以显著提升性能。

## What Changes

### New Capabilities
- `spmv-kernels` - SpMV CUDA 内核实现

### Modified Capabilities
- `csr-format` - 添加 SpMV CSR 内核
- `ell-format` - 添加 SpMV ELL 内核

## Impact

**New Files:**
- `include/spmv/spmv.h` - SpMV 接口头文件
- `src/spmv_kernels.cu` - CUDA 内核实现
- `src/spmv_cpu.cpp` - CPU 参考实现
- `tests/test_spmv.cu` - SpMV 测试

**Kernel Types:**
- `SCALAR_CSR` - 每个线程处理一行
- `VECTOR_CSR` - 每个 warp 处理一行
- `MERGE_PATH` - 负载均衡分区
- `ELL_KERNEL` - ELL 格式专用内核

**Performance Targets:**
- 相对误差 < 1e-6 (单精度)
- 带宽利用率 > 60% 理论峰值

## Status

✅ Completed - 2025-02-20
