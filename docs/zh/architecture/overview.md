# 架构概览

GPU SpMV 采用分层架构设计，清晰分离存储、计算和应用层。

## 系统架构

```mermaid
graph TB
    subgraph Application["应用层"]
        PR[PageRank]
        IS[迭代求解器]
        GNN[图神经网络]
        SC[科学计算]
    end

    subgraph API["API 层"]
        spmv_csr[spmv_csr]
        spmv_ell[spmv_ell]
        benchmark[benchmark]
        pagerank[pagerank]
    end

    subgraph Kernel["Kernel 层"]
        Scalar["Scalar CSR"]
        Vector["Vector CSR"]
        Merge["Merge Path"]
        ELL["ELL Kernel"]
    end

    subgraph Storage["存储层"]
        CSR_M["CSR Matrix"]
        ELL_M["ELL Matrix"]
    end

    Application --> API
    API --> Kernel
    Kernel --> Storage
```

## 设计原则

| 原则 | 实现方式 | 好处 |
|:-----|:---------|:-----|
| 分层架构 | 存储、计算、应用分离 | 关注点分离，易于维护 |
| 策略模式 | Kernel 选择可插拔 | 灵活扩展新算法 |
| RAII 管理 | CudaBuffer 自动释放 | 防止内存泄漏 |
| 错误语义化 | SpMVError 枚举 | 清晰诊断信息 |

## 四层架构

### 存储层

定义稀疏矩阵的内存布局：

- **CSR Matrix** — 通用格式，存储高效
- **ELL Matrix** — 列优先存储，GPU 优化

### Kernel 层

实现四种优化的 SpMV 内核：

| Kernel | 线程策略 | 最佳场景 | 带宽效率 |
|:-------|:---------|:---------|:--------:|
| Scalar CSR | 1 线程/行 | 极稀疏 (nnz/row < 4) | ~40-50% |
| Vector CSR | 1 Warp/行 | 均匀分布 | ~65-75% |
| Merge Path | 动态分块 | 高度倾斜 | ~70-80% |
| ELL Kernel | 列并行 | 行长度均匀 | ~80-90% |

### API 层

提供用户友好的接口：

- `spmv_csr()` — CSR 格式 SpMV
- `spmv_ell()` — ELL 格式 SpMV
- `spmv_auto_config()` — 自动选择最优 Kernel
- `pagerank()` — PageRank 算法

### 应用层

构建在 SpMV 之上的应用：

- **PageRank** — 网页排名算法
- **迭代求解器** — CG、GMRES 等
- **图神经网络** — 稀疏图卷积
- **科学计算** — 有限元、CFD

## 设计亮点

### 1. RAII 资源管理

```cpp
// 自动生命周期管理，防止内存泄漏
class CudaBuffer {
public:
    explicit CudaBuffer(size_t n) { cudaMalloc(&ptr_, n * sizeof(T)); }
    ~CudaBuffer() { cudaFree(ptr_); }
    // 禁用拷贝，允许移动
};
```

### 2. 执行上下文

```cpp
// 缓存纹理对象，避免重复创建
SpMVExecutionContext ctx;
for (int i = 0; i < n_iter; i++) {
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
    // 纹理对象被复用
}
```

### 3. 自动 Kernel 选择

```cpp
// 基于矩阵特征自动选择最优 Kernel
SpMVConfig config = spmv_auto_config(csr);
```

## 相关文档

- [Kernel 选择策略](/zh/architecture/kernel-selection)
- [内存布局](/zh/architecture/memory-layout)
- [Spec-Driven 开发](/zh/architecture/spec-driven)
