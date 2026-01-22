# GPU SpMV (稀疏矩阵向量乘法)

基于 CUDA 的高性能稀疏矩阵向量乘法库，支持 CSR 和 ELL 格式，包含多种负载均衡优化策略。

## 特性

- **多种稀疏矩阵格式**
  - CSR (Compressed Sparse Row)
  - ELL (ELLPACK)

- **优化的 CUDA Kernels**
  - Scalar CSR: 一个线程处理一行
  - Vector CSR: 一个 Warp (32线程) 处理一行
  - Merge Path: 工作量均匀分配，适合高度不均匀的矩阵
  - ELL Kernel: Column-major 访问，适合均匀行长度

- **自动 Kernel 选择**
  - 基于矩阵特征自动选择最优 Kernel

- **性能度量**
  - 带宽利用率分析
  - GFLOPS 计算
  - 完整的基准测试框架
  - 可选纹理缓存读取输入向量

- **应用示例**
  - PageRank 图算法实现

## 构建

### 要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- NVIDIA GPU (Compute Capability 7.0+)

### 编译

```bash
mkdir build
cd build
cmake ..
make -j
```

### 运行测试

```bash
./spmv_tests
```

### 验证路径

```bash
# 运行属性测试与单元测试
./spmv_tests

# 运行基准测试程序
./benchmarks_main
```

## 使用示例

### 基础 SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

// 创建稠密矩阵
std::vector<float> dense = {
    1, 0, 2,
    0, 3, 4,
    0, 0, 5
};

// 转换为 CSR 格式
CSRMatrix* csr = csr_create(0, 0, 0);
csr_from_dense(csr, dense.data(), 3, 3);
csr_to_gpu(csr);

// 输入向量
std::vector<float> x = {1, 1, 1};
CudaBuffer<float> d_x(3);
CudaBuffer<float> d_y(3);
d_x.copyFromHost(x.data(), 3);

// 执行 SpMV
SpMVConfig config = spmv_auto_config(csr);
// config.use_texture = true;  // 可选：启用纹理缓存读取 x
SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

// 获取结果
std::vector<float> y(3);
d_y.copyToHost(y.data(), 3);

csr_destroy(csr);
```

### PageRank

```cpp
#include "spmv/pagerank.h"

// 创建邻接矩阵 (列归一化)
CSRMatrix* adj = /* ... */;
csr_to_gpu(adj);

// 运行 PageRank
PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

PageRankResult result = pagerank(adj, &config);

// 获取 Top-10 节点
std::vector<TopKNode> top_10(10);
pagerank_top_k(&result, adj->num_rows, 10, top_10.data());

pagerank_free(&result);
csr_destroy(adj);
```

### 基准测试

```cpp
#include "spmv/benchmark.h"

BenchmarkConfig bench_config;
bench_config.num_warmup_runs = 5;
bench_config.num_runs = 20;

BenchmarkResult result = benchmark_csr(csr, x.data(), &spmv_config, &bench_config);

std::cout << "Avg time: " << result.avg_time_ms << " ms\n";
std::cout << "GFLOPS: " << result.gflops << "\n";
std::cout << "Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";

// 导出 JSON
std::string json = benchmark_to_json(result);
```

## 性能优化

### Kernel 选择策略

- **短行 (avg_nnz < 4)**: Scalar CSR
- **均匀分布 (skewness < 10)**: Vector CSR
- **高度不均匀 (skewness >= 10)**: Merge Path

### 带宽优化

- Column-major 存储 (ELL 格式)
- 合并内存访问
- 带宽利用率监控

## 测试

项目包含完整的属性测试套件，验证：

- CSR/ELL 格式转换正确性
- SpMV 计算正确性 (与 CPU 参考对比)
- 维度验证
- Kernel 选择器有效性
- 带宽度量有效性
- PageRank 不变量

每个属性测试运行 100 次迭代，使用随机生成的矩阵。

## 项目结构

```
.
├── include/spmv/       # 头文件
│   ├── common.h
│   ├── cuda_buffer.h
│   ├── csr_matrix.h
│   ├── ell_matrix.h
│   ├── spmv.h
│   ├── bandwidth.h
│   ├── benchmark.h
│   └── pagerank.h
├── src/                # 源文件
│   ├── csr_matrix.cpp
│   ├── ell_matrix.cpp
│   ├── spmv_cpu.cpp
│   ├── spmv_kernels.cu
│   ├── bandwidth.cpp
│   ├── benchmark.cu
│   └── pagerank.cu
├── tests/              # 测试文件
└── benchmarks/         # 基准测试程序
```

## 许可证

MIT License
