-- -layout : default title : 架构设计 nav_order : 6 permalink : / architecture lang : zh-- -

    <p align = "right"><a href = "architecture.en">🇺🇸 English</ a></ p>
#架构设计
    { :.no_toc}

    系统架构、核心算法和设计决策。 {: .fs-6 .fw-300
}

##目录 {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

    ##系统架构

```
┌─────────────────────────────────────────────────────────┐
│ 应用层                              │
│ PageRank  │ 迭代求解器  │ 图神经网络  │ 科学计算    │
├─────────────────────────────────────────────────────────┤
│ API 层                              │
│ spmv_csr  │ spmv_ell  │ benchmark  │ pagerank    │
├─────────────────────────────────────────────────────────┤
│ Kernel 层                           │
│ Scalar CSR │ Vector CSR │ Merge Path │ ELL Kernel   │
├─────────────────────────────────────────────────────────┤
│ 存储层                              │
│ CSR Matrix      │ ELL Matrix        │
└─────────────────────────────────────────────────────────┘
```

        **设计原则 ** : -**分层架构 ** : 清晰分离存储、计算和应用层 -
    **策略模式 ** : Kernel 选择策略可插拔 -
    **RAII 管理 ** : 自动资源生命周期管理

                     -- -

    ##稀疏矩阵格式

    ## #CSR(Compressed Sparse Row)

``` 原始矩阵 : CSR 存储 :
┌─────┬─────┐ values : [ 1, 2, 3, 4, 5 ]
│ 1 0 2 │ col_indices : [ 0, 2, 1, 2, 3 ]
│ 0 3 4 │ = > row_ptrs : [ 0, 2, 4, 5 ]
│ 0 0 5 │     
└─────┴─────┘ row_ptrs[i] 表示第 i 行的起始位置
```

                              **特点 ** : -通用格式，存储高效
                                          -
                                          适合不规则稀疏模式 -
                                          行遍历效率高

                                              ## #ELL(ELLPACK)

``` 原始矩阵 : ELL 存储(列优先)
    :
┌─────┬─────┐ values : [ 1, 3, 5, 2, 4, 0 ]
│ 1 0 2 │ col_indices
    : [ 0, 1, 3, 2, 2, -]
│ 0 3 4 │     
│ 0 0 5 │ 每行补齐到最大长度，支持完全合并访存
└─────┴─────┘
```

      *
      *特点 * * : -列优先存储，合并访存 - 适合行长度均匀的矩阵
                                          -
                                          GPU 上性能最佳

                                          -- -

                                          ##Kernel 选择策略

``` 输入 : CSR 矩阵
       │
       ├── avg_nnz_per_row <
                 4 ──→ Scalar CSR
       │ (单线程处理，最小开销)
       │
       └── avg_nnz_per_row >= 4
               │
               ├── 偏度 < 10 ──→ Vector CSR
               │ (Warp 协作，均衡负载)
               │
               └── 偏度 >= 10 ──→ Merge Path(完美负载均衡)
```

                 ## #Kernel 对比

             | Kernel | 线程策略 | 最佳场景 | 带宽效率 | | : -- -- -- - | : -- -- -- -- - |
    : -- -- -- -- - | : -- -- -- -- : | | Scalar CSR | 1 线程 / 行 | 极稀疏 | 中等 | | Vector CSR |
                                      1 warp / 行 | 均匀分布 | 高 |
                                      | Merge Path | 动态分块 | 高度倾斜 | 最高 |
                                      | ELL | 列并行 | 均匀行长度 | 极高 |

                                      -- -

                                         ##设计决策

                                         ## #1. RAII 资源管理

```cpp
                                         // 自动生命周期管理，防止内存泄漏
                                         class CudaBuffer {
   public:
    explicit CudaBuffer(size_t n) { cudaMalloc(&ptr_, n * sizeof(T)); }
    ~CudaBuffer() { cudaFree(ptr_); }
    // 禁用拷贝，允许移动
};
```

    ## #2. 执行上下文

```cpp
        // 缓存纹理对象，避免重复创建
        SpMVExecutionContext ctx;
for (int i = 0; i < n_iter; i++) {
    spmv_csr(csr, d_x, d_y, &config, n, &ctx);
    // 纹理对象被复用
}
```

    ## #3. 错误处理

```cpp
    // 语义化错误码
    enum class SpMVError {
        SUCCESS = 0,
        INVALID_DIMENSION = -1,
        CUDA_MALLOC = -2,
        // ...
    };
```

    -- -

    <div class = "text-center text-small" style = "margin-top: 3rem;">
    <p> 更多技术细节见<a href = "https://github.com/LessUp/gpu-spmv/tree/master/openspec/specs">
        openspec / specs / </ a> 目录</ p></ div>
