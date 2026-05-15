# 常见问题

## 安装与配置

### CUDA 版本要求？

GPU SpMV 需要以下 CUDA 版本：

| CUDA 版本 | 支持的 GPU 架构 |
|:---------:|:---------------|
| 11.0+ | Volta (SM 7.0), Turing (SM 7.5) |
| 11.1+ | Ampere (SM 8.0, 8.6) |
| 11.8+ | Ada Lovelace (SM 8.9) |
| 12.0+ | Hopper (SM 9.0) |

**推荐**: CUDA 12.0+ 以获得最佳性能和兼容性。

::: tip 无 GPU 环境
使用 `-DSPMV_REQUIRE_CUDA=OFF` 可在无 GPU 环境下构建 CPU-only 版本：
```bash
cmake -S . -B build -DSPMV_REQUIRE_CUDA=OFF
cmake --build build
```
:::

### 支持哪些操作系统？

- **Linux**: Ubuntu 20.04+, CentOS 7+, Debian 10+
- **Windows**: Windows 10/11 with Visual Studio 2019+
- **macOS**: 不支持（无 NVIDIA GPU）

### 如何验证安装成功？

运行测试套件：

```bash
cd build
ctest --preset default
```

所有测试通过即表示安装成功。

---

## 矩阵格式

### CSR 和 ELL 格式如何选择？

| 格式 | 适用场景 | 性能特点 |
|:-----|:---------|:---------|
| **CSR** | 通用稀疏矩阵，非零元素分布不规则 | 适用于大多数场景，自动内核选择 |
| **ELL** | 每行非零元素数量相近 | 完全合并访存，最高带宽利用率 |

**建议**: 默认使用 CSR，对于均匀矩阵可转换为 ELL 以获得更高性能。

### 如何转换矩阵格式？

```cpp
#include <spmv/spmv.h>

// 从 CSR 转换为 ELL
CSRMatrix* csr = csr_create(rows, cols, nnz);
// ... 填充 CSR ...

ELLMatrix* ell = ell_create(rows, cols, max_nnz_per_row);
ell_from_csr(ell, csr);  // 自动转换
```

---

## 性能优化

### 为什么我的性能低于 70%？

可能的原因和解决方案：

1. **矩阵规模太小**
   - 问题：GPU 未充分利用
   - 解决：矩阵规模建议 > 10K × 10K

2. **非零元素分布极度不均匀**
   - 问题：负载不均衡
   - 解决：Merge Path 内核会自动处理，或尝试调整矩阵结构

3. **GPU 架构较旧**
   - 问题：缺少现代 GPU 特性
   - 解决：使用 Compute Capability 7.0+ 的 GPU

4. **内存带宽限制**
   - 问题：其他进程占用 GPU 内存
   - 解决：确保 GPU 内存充足，关闭其他 GPU 应用

### 如何选择最优内核？

使用 `spmv_auto_config()` 自动选择：

```cpp
SpMVConfig config = spmv_auto_config(csr);
// 自动根据矩阵特征选择最优内核
```

选择策略：
- `avg_nnz_per_row < 4` → Scalar CSR
- `skewness < 10` → Vector CSR
- `skewness >= 10` → Merge Path

### 如何复用配置以提高批量操作性能？

```cpp
// 只计算一次配置
SpMVConfig config = spmv_auto_config(csr);

// 复用配置进行多次 SpMV
for (int i = 0; i < iterations; i++) {
    spmv_csr(csr, x[i], y[i], &config, n);
}
```

---

## 与其他库对比

### 与 cuSPARSE 相比如何？

| 特性 | GPU SpMV | cuSPARSE |
|:-----|:--------:|:--------:|
| 开源 | ✅ | ❌ |
| 自动内核选择 | ✅ | ❌ |
| Merge Path 算法 | ✅ | ❌ |
| ELL 格式支持 | ✅ | ✅ |
| 不规则矩阵性能 | **更优** | 一般 |
| 均匀矩阵性能 | 相近 | 相近 |

### 与其他开源库相比？

| 库 | Stars | 特点 |
|:---|:-----:|:-----|
| **GPU SpMV** | - | 自动选择，Merge Path，完整文档 |
| [Ginkgo](https://github.com/ginkgo-project/ginkgo) | 597 | 多后端，性能可移植 |
| [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) | 300+ | 性能可移植，多平台 |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | N/A | 官方，多格式 |

---

## 故障排除

### 编译错误：CUDA not found

确保 CUDA 安装正确：

```bash
# 检查 CUDA 版本
nvcc --version

# 设置 CUDA 路径（如果需要）
export CUDA_HOME=/usr/local/cuda
```

### 运行时错误：invalid device ordinal

GPU 设备编号错误：

```cpp
// 检查可用 GPU 数量
int device_count;
cudaGetDeviceCount(&device_count);

// 设置正确的设备
cudaSetDevice(0);  // 使用第一个 GPU
```

### 性能测试结果不稳定

确保：
1. GPU 温度正常（避免过热降频）
2. 无其他 GPU 进程干扰
3. 预热后再测试（warmup iterations）

```cpp
// 预热
for (int i = 0; i < 10; i++) {
    spmv_csr(csr, x, y, &config, n);
}

// 正式测试
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 100; i++) {
    spmv_csr(csr, x, y, &config, n);
}
auto end = std::chrono::high_resolution_clock::now();
```

---

## 更多问题？

如果以上内容没有解决您的问题：

1. 查看 [API 参考](/zh/api/spmv) 了解详细用法
2. 查看 [性能指南](/zh/performance/optimization-guide) 了解优化技巧
3. 在 [GitHub Issues](https://github.com/AICL-Lab/gpu-spmv/issues) 提问
