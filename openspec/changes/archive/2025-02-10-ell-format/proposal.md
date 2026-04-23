# Add ELL Format Support

## Why

需要支持稀疏矩阵的 ELL (ELLPACK) 格式存储，以优化 GPU 内存合并访问。ELL 格式对于行长度均匀的矩阵特别高效，因为其列主存储布局可以实现完全合并的内存访问。

## What Changes

### New Capabilities
- `ell-format` - ELL 稀疏矩阵存储格式

### Modified Capabilities
- `csr-format` - 添加 CSR 到 ELL 格式转换

## Impact

**New Files:**
- `include/spmv/ell_matrix.h` - ELL 矩阵头文件
- `src/ell_matrix.cpp` - ELL 矩阵实现
- `tests/test_ell.cpp` - ELL 单元测试

**API Functions:**
- `ell_create()` - 创建 ELL 矩阵
- `ell_destroy()` - 销毁 ELL 矩阵
- `ell_from_dense()` - 从稠密矩阵转换
- `ell_from_csr()` - 从 CSR 格式转换
- `ell_to_gpu()` - 传输到 GPU
- `ell_serialize()` / `ell_deserialize()` - 序列化

## Status

✅ Completed - 2025-02-10
