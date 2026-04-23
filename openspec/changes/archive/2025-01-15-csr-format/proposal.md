# Add CSR Format Support

## Why

需要支持稀疏矩阵的 CSR (Compressed Sparse Row) 格式存储，以便高效进行 GPU 加速的 SpMV 运算。CSR 格式是稀疏矩阵最常用的存储格式之一，适用于通用稀疏矩阵运算。

## What Changes

### New Capabilities
- `csr-format` - CSR 稀疏矩阵存储格式

### Modified Capabilities
- None (initial implementation)

## Impact

**New Files:**
- `include/spmv/csr_matrix.h` - CSR 矩阵头文件
- `src/csr_matrix.cpp` - CSR 矩阵实现
- `tests/test_csr.cpp` - CSR 单元测试

**API Functions:**
- `csr_create()` - 创建 CSR 矩阵
- `csr_destroy()` - 销毁 CSR 矩阵
- `csr_from_dense()` - 从稠密矩阵转换
- `csr_to_gpu()` - 传输到 GPU
- `csr_get_element()` - 元素查询
- `csr_serialize()` / `csr_deserialize()` - 序列化

## Status

✅ Completed - 2025-01-15
