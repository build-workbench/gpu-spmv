# Implementation Plan: GPU SpMV (稀疏矩阵向量乘法)

## Overview

本实现计划将设计分解为增量式编码任务，从基础数据结构开始，逐步构建 CUDA Kernel、优化策略和应用层。每个任务都可独立验证，确保渐进式开发。

## Tasks

- [x] 1. 项目结构和基础设施
  - [x] 1.1 创建项目目录结构和 CMake 构建系统
    - 创建 `src/`, `include/`, `tests/`, `benchmarks/` 目录
    - 配置 CMake 支持 CUDA 编译
    - 设置 Google Test 依赖
    - _Requirements: 8.4_

  - [x] 1.2 实现错误处理基础设施
    - 定义 `SpMVError` 枚举和错误字符串函数
    - 实现 `CUDA_CHECK` 宏
    - 实现 `CudaBuffer<T>` RAII 模板类
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 1.3 编写错误处理单元测试
    - 测试 CudaBuffer 构造和析构
    - 测试错误码字符串转换
    - _Requirements: 8.4, 8.5_


- [x] 2. CSR 格式存储实现
  - [x] 2.1 实现 CSR 数据结构和基本操作
    - 实现 `CSRMatrix` 结构体
    - 实现 `csr_create`, `csr_destroy` 函数
    - 实现 `csr_from_dense` 稠密矩阵转换
    - 实现 `csr_get_element` 元素查询
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 实现 CSR GPU 内存传输
    - 实现 `csr_to_gpu` 主机到设备传输
    - 实现 `csr_from_gpu` 设备到主机传输
    - _Requirements: 1.1, 1.4_

  - [x] 2.3 实现 CSR 序列化
    - 实现 `csr_serialize` 二进制写入
    - 实现 `csr_deserialize` 二进制读取
    - _Requirements: 1.5_

  - [x] 2.4 编写 CSR 属性测试
    - **Property 1: CSR Dense-to-Sparse Round Trip**
    - **Property 2: CSR Element Lookup Correctness**
    - **Property 3: CSR Serialization Round Trip**
    - **Validates: Requirements 1.2, 1.3, 1.5**

- [x] 3. ELL 格式存储实现
  - [x] 3.1 实现 ELL 数据结构和基本操作
    - 实现 `ELLMatrix` 结构体
    - 实现 `ell_create`, `ell_destroy` 函数
    - 实现 `ell_from_dense` 稠密矩阵转换
    - 实现 `ell_from_csr` CSR 转换
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.2 实现 ELL GPU 内存传输和序列化
    - 实现 `ell_to_gpu` 传输函数
    - 实现 `ell_serialize`, `ell_deserialize`
    - _Requirements: 2.5_

  - [x] 3.3 编写 ELL 属性测试
    - **Property 4: ELL Dense-to-Sparse Round Trip**
    - **Property 5: ELL Padding Correctness**
    - **Property 6: ELL Column-Major Layout**
    - **Property 7: ELL Serialization Round Trip**
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5**


- [ ] 4. Checkpoint - 存储层验证
  - 确保所有存储层测试通过
  - 验证 CSR 和 ELL 格式转换正确性
  - 如有问题请询问用户

- [x] 5. 基础 SpMV Kernel 实现
  - [x] 5.1 实现 CPU 参考 SpMV
    - 实现 `spmv_cpu_csr` 作为正确性基准
    - 实现 `spmv_cpu_ell` 作为正确性基准
    - _Requirements: 3.1, 3.2_

  - [x] 5.2 实现 Scalar CSR Kernel
    - 实现 `spmv_csr_scalar` CUDA kernel
    - 一个线程处理一行
    - 实现维度验证和错误处理
    - _Requirements: 3.1, 3.3, 3.4, 3.5_

  - [x] 5.3 实现 ELL Kernel
    - 实现 `spmv_ell` CUDA kernel
    - Column-major 访问模式
    - _Requirements: 3.2, 3.3_

  - [x] 5.4 编写基础 SpMV 属性测试
    - **Property 8: SpMV CSR Correctness**
    - **Property 9: SpMV ELL Correctness**
    - **Property 10: SpMV Dimension Validation**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.5**

- [x] 6. 负载均衡 Kernel 实现
  - [x] 6.1 实现 Vector CSR Kernel
    - 实现 `spmv_csr_vector` CUDA kernel
    - 一个 Warp (32线程) 处理一行
    - 使用 `__shfl_down_sync` 进行 Warp 级归约
    - _Requirements: 4.1, 4.2_

  - [x] 6.2 实现 Merge Path Kernel
    - 实现 `merge_path_search` 设备函数
    - 实现 `spmv_csr_merge_path` CUDA kernel
    - 工作量均匀分配到所有线程
    - _Requirements: 4.3, 4.4_

  - [x] 6.3 实现 Kernel 选择器
    - 实现 `spmv_auto_config` 函数
    - 基于矩阵特征选择最优 Kernel
    - _Requirements: 4.5_

  - [x] 6.4 编写 Kernel 选择器属性测试
    - **Property 11: Kernel Selector Validity**
    - **Validates: Requirements 4.5**


- [x] 7. Checkpoint - 计算层验证
  - 确保所有 SpMV Kernel 测试通过
  - 验证不同 Kernel 结果一致性
  - 如有问题请询问用户

- [x] 8. 带宽优化实现
  - [x] 8.1 实现纹理内存优化
    - 绑定输入向量 x 到纹理内存
    - 实现 `fetch_x` 设备函数
    - 修改 Kernel 支持纹理访问
    - _Requirements: 5.3_

  - [x] 8.2 实现带宽度量
    - 实现 `BandwidthMetrics` 结构体
    - 实现 `compute_bandwidth` 函数
    - 在 SpMV 结果中返回带宽指标
    - _Requirements: 5.1, 5.5_

  - [x] 8.3 编写带宽度量属性测试
    - **Property 12: Bandwidth Metrics Validity**
    - **Validates: Requirements 5.5**

- [ ] 9. 基准测试套件
  - [x] 9.1 实现基准测试框架
    - 实现 `BenchmarkResult` 结构体
    - 实现多次运行统计 (avg, min, max, stddev)
    - 实现 GPU vs CPU 对比
    - _Requirements: 6.1, 6.3, 6.4_

  - [x] 9.2 实现 JSON 报告生成
    - 实现 `benchmark_to_json` 函数
    - 实现 `benchmark_from_json` 解析
    - _Requirements: 6.5_

  - [x] 9.3 编写基准测试属性测试
    - **Property 13: Benchmark Metrics Completeness**
    - **Property 14: Benchmark JSON Round Trip**
    - **Validates: Requirements 6.1, 6.3, 6.5**


- [x] 10. Checkpoint - 核心功能完成
  - 确保所有核心 SpMV 功能测试通过
  - 验证带宽度量和基准测试正常工作
  - 如有问题请询问用户

- [x] 11. PageRank 图算法实现 (可选扩展)
  - [x] 11.1 实现 PageRank 核心算法
    - 实现 `PageRankConfig` 和 `PageRankResult` 结构体
    - 实现迭代 SpMV 计算 PageRank
    - 实现收敛检测 (L2 范数)
    - _Requirements: 7.1, 7.2_

  - [x] 11.2 实现悬挂节点处理
    - 实现 `handle_dangling_nodes` 函数
    - 正确处理无出边节点
    - _Requirements: 7.3_

  - [x] 11.3 实现 Top-K 结果输出
    - 实现排序和 Top-K 选择
    - _Requirements: 7.5_

  - [x] 11.4 编写 PageRank 属性测试
    - **Property 15: PageRank Score Invariants**
    - **Property 16: PageRank Top-K Ordering**
    - **Validates: Requirements 7.1, 7.2, 7.5**

- [ ] 12. 最终 Checkpoint
  - 确保所有测试通过
  - 运行完整基准测试套件
  - 验证 PageRank 示例正常工作
  - 如有问题请询问用户

## Notes

- 所有任务（包括测试任务）都是必须执行的
- 每个属性测试运行至少 100 次迭代
- Checkpoint 任务用于阶段性验证，确保增量开发质量
- 建议按顺序执行任务，因为后续任务依赖前置任务
