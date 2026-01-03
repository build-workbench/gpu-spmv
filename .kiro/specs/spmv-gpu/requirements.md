# Requirements Document

## Introduction

本项目实现基于 GPU 的稀疏矩阵向量乘法 (SpMV) 算子，支持 CSR (Compressed Sparse Row) 和 ELL 格式的稀疏矩阵存储。项目重点解决 GPU 处理稀疏数据时的核心挑战：不规则内存访问、负载不均衡和带宽利用率优化。可选扩展实现 PageRank 图算法作为应用示例。

## Glossary

- **SpMV**: Sparse Matrix-Vector Multiplication，稀疏矩阵向量乘法
- **CSR**: Compressed Sparse Row，压缩稀疏行格式，使用三个数组存储稀疏矩阵
- **ELL**: ELLPACK 格式，将每行填充到相同长度的稀疏矩阵存储格式
- **Warp**: GPU 中 32 个线程组成的执行单元
- **Warp_Divergence**: 线程束分歧，同一 Warp 内线程执行不同分支导致的性能损失
- **Load_Balancing**: 负载均衡，确保各线程工作量均匀分配
- **Bandwidth_Bound**: 带宽受限，计算性能受限于内存带宽而非计算能力
- **Coalesced_Access**: 合并访问，相邻线程访问连续内存地址以最大化带宽
- **Dense_Matrix**: 稠密矩阵，大部分元素非零的矩阵
- **Sparse_Matrix**: 稀疏矩阵，大部分元素为零的矩阵
- **PageRank**: 网页排名算法，基于迭代矩阵向量乘法

## Requirements

### Requirement 1: CSR 格式稀疏矩阵存储

**User Story:** As a developer, I want to store sparse matrices in CSR format, so that I can efficiently represent matrices with many zero elements using minimal memory.

#### Acceptance Criteria

1. THE CSR_Storage SHALL store a sparse matrix using three arrays: values (非零元素值), column_indices (列索引), and row_pointers (行指针)
2. WHEN a dense matrix is converted to CSR format, THE CSR_Storage SHALL preserve all non-zero elements and their positions exactly
3. WHEN querying an element at position (i, j), THE CSR_Storage SHALL return the correct value (non-zero value or zero)
4. THE CSR_Storage SHALL support matrices with up to 10 million non-zero elements
5. WHEN serializing CSR data to disk, THE CSR_Storage SHALL encode it using a binary format that can be deserialized back to an equivalent CSR structure

### Requirement 2: ELL 格式稀疏矩阵存储

**User Story:** As a developer, I want to store sparse matrices in ELL format, so that I can achieve better memory coalescing on GPU for matrices with uniform row lengths.

#### Acceptance Criteria

1. THE ELL_Storage SHALL store a sparse matrix using two 2D arrays: values and column_indices, with each row padded to max_nnz_per_row
2. WHEN a dense matrix is converted to ELL format, THE ELL_Storage SHALL preserve all non-zero elements and their positions exactly
3. WHEN a row has fewer non-zero elements than max_nnz_per_row, THE ELL_Storage SHALL pad with zeros and invalid column indices (-1)
4. THE ELL_Storage SHALL store data in column-major order for coalesced GPU memory access
5. WHEN serializing ELL data to disk, THE ELL_Storage SHALL encode it using a binary format that can be deserialized back to an equivalent ELL structure

### Requirement 3: 基础 SpMV CUDA Kernel

**User Story:** As a developer, I want a basic SpMV CUDA kernel, so that I can perform sparse matrix-vector multiplication on GPU.

#### Acceptance Criteria

1. WHEN executing SpMV with CSR format, THE SpMV_Kernel SHALL compute y = A * x where A is the sparse matrix and x is the input vector
2. WHEN executing SpMV with ELL format, THE SpMV_Kernel SHALL compute y = A * x correctly
3. THE SpMV_Kernel SHALL produce results within 1e-6 relative error compared to CPU reference implementation for single precision
4. THE SpMV_Kernel SHALL handle matrices where some rows have zero non-zero elements
5. IF the input vector dimension does not match matrix column count, THEN THE SpMV_Kernel SHALL return an error code

### Requirement 4: 负载均衡优化

**User Story:** As a developer, I want load-balanced SpMV kernels, so that I can avoid performance degradation from uneven row lengths.

#### Acceptance Criteria

1. THE Vector_CSR_Kernel SHALL assign one warp (32 threads) to process each matrix row, with threads cooperatively processing non-zero elements
2. WHEN processing rows with varying lengths, THE Vector_CSR_Kernel SHALL achieve at least 80% of peak performance compared to uniform-length matrices
3. THE Merge_Path_Kernel SHALL partition work evenly across all threads regardless of row length distribution
4. WHEN the matrix has highly skewed row lengths (max/min ratio > 100), THE Merge_Path_Kernel SHALL maintain at least 70% efficiency compared to uniform distribution
5. THE Load_Balanced_Kernels SHALL provide a kernel selection function that chooses optimal kernel based on matrix characteristics

### Requirement 5: 带宽优化

**User Story:** As a developer, I want bandwidth-optimized SpMV implementation, so that I can maximize GPU memory throughput for bandwidth-bound operations.

#### Acceptance Criteria

1. THE Optimized_SpMV SHALL achieve at least 60% of theoretical peak memory bandwidth on target GPU
2. WHEN accessing matrix data, THE Optimized_SpMV SHALL use coalesced memory access patterns where possible
3. THE Optimized_SpMV SHALL support texture memory caching for the input vector x to improve cache hit rate
4. WHEN the matrix fits in L2 cache, THE Optimized_SpMV SHALL demonstrate improved performance from cache reuse
5. THE Optimized_SpMV SHALL provide bandwidth utilization metrics after each SpMV operation

### Requirement 6: 性能基准测试

**User Story:** As a developer, I want comprehensive benchmarking tools, so that I can measure and compare SpMV performance across different implementations.

#### Acceptance Criteria

1. THE Benchmark_Suite SHALL measure execution time, GFLOPS, and bandwidth utilization for each SpMV kernel
2. THE Benchmark_Suite SHALL support standard sparse matrix test sets (e.g., matrices from SuiteSparse collection)
3. WHEN running benchmarks, THE Benchmark_Suite SHALL report average, min, max, and standard deviation over multiple runs
4. THE Benchmark_Suite SHALL compare GPU implementation against CPU baseline
5. THE Benchmark_Suite SHALL generate performance reports in JSON format for analysis

### Requirement 7: PageRank 图算法实现 (可选扩展)

**User Story:** As a developer, I want to implement PageRank using SpMV, so that I can demonstrate practical application of sparse matrix operations on graph data.

#### Acceptance Criteria

1. WHEN given an adjacency matrix and damping factor, THE PageRank_Algorithm SHALL compute page rank scores using iterative SpMV
2. THE PageRank_Algorithm SHALL converge when the L2 norm of rank difference between iterations is below 1e-6
3. WHEN the graph has dangling nodes (nodes with no outgoing edges), THE PageRank_Algorithm SHALL handle them correctly
4. THE PageRank_Algorithm SHALL support graphs with up to 1 million nodes
5. THE PageRank_Algorithm SHALL output the top-k nodes by rank score

### Requirement 8: 错误处理与资源管理

**User Story:** As a developer, I want robust error handling and resource management, so that I can safely use the SpMV library in production code.

#### Acceptance Criteria

1. IF CUDA memory allocation fails, THEN THE SpMV_Library SHALL return a descriptive error and release any partially allocated resources
2. IF kernel launch fails, THEN THE SpMV_Library SHALL capture the CUDA error and propagate it to the caller
3. WHEN SpMV operations complete, THE SpMV_Library SHALL properly synchronize and check for asynchronous errors
4. THE SpMV_Library SHALL provide RAII-style resource management for GPU memory allocations
5. IF invalid matrix dimensions are provided, THEN THE SpMV_Library SHALL validate inputs and return appropriate error codes before any GPU operations
