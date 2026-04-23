# Add Performance Benchmarking Suite

## Why

需要全面的性能基准测试工具来测量和比较不同 SpMV 实现的性能，以便进行性能优化和验证。

## What Changes

### New Capabilities
- `benchmark` - 性能基准测试框架

### Modified Capabilities
- None (独立模块)

## Impact

**New Files:**
- `include/spmv/benchmark.h` - 基准测试接口
- `src/benchmark.cu` - 基准测试实现
- `benchmarks/main.cu` - 基准测试可执行文件
- `tests/test_benchmark.cu` - 基准测试验证

**Features:**
- 多次运行统计 (avg, min, max, stddev)
- GFLOPS 和带宽利用率计算
- GPU vs CPU 性能对比
- JSON 格式报告导出
- 支持 SuiteSparse 矩阵集合

## Status

✅ Completed - 2025-03-05
