# 更新日志

GPU SpMV 的所有重要变更记录于此。

## [1.0.0] - 2026-04-01

### 新增
- 完整的 SpMV 实现，包含 4 种优化 Kernel
- CSR 和 ELL 稀疏矩阵格式
- 基于矩阵统计的自动 Kernel 选择
- PageRank 算法实现
- 完整的基准测试套件
- RAII 内存管理 (CudaBuffer)
- 语义化错误码 (SpMVError)
- OpenSpec 规范驱动开发

### 性能
- RTX 3090 上 70%+ 内存带宽利用率
- Merge Path Kernel 实现负载均衡
- ELL Kernel 实现合并访存
- 大向量纹理缓存支持

### 文档
- 双语文档（中文/英文）
- 完整 API 参考
- 架构设计文档
- 学术参考

## [0.9.0] - 2025-03-10

### 新增
- PageRank 应用层
- Top-K 节点提取工具

## [0.8.0] - 2025-03-05

### 新增
- 带统计的基准测试框架
- JSON 导出基准测试结果

## [0.7.0] - 2025-03-01

### 新增
- 自动 Kernel 选择 (`spmv_auto_config`)
- 可配置的选择阈值

## [0.6.0] - 2025-02-20

### 新增
- 面向倾斜矩阵的 Merge Path Kernel
- 矩阵统计计算

### 性能
- 改进不规则矩阵的负载均衡

## [0.5.0] - 2025-02-10

### 新增
- ELL 矩阵格式
- 面向均匀矩阵的 ELL Kernel
- CSR 到 ELL 转换

## [0.4.0] - 2025-01-15

### 新增
- CSR 矩阵格式
- Scalar CSR Kernel
- Vector CSR Kernel
- 基础 SpMV 计算

## [0.1.0] - 2024-12-01

### 新增
- 初始项目结构
- CMake 构建系统
- Google Test 集成
