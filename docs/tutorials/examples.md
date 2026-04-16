---
layout: default
title: 示例代码
parent: 中文文档
nav_order: 4
lang: zh
---

# 📝 示例代码
{: .no_toc }

完整的代码示例，从基础用法到高级应用。
{: .fs-6 .fw-300 }

## 基础示例

### 最简 SpMV

```cpp
#include "spmv/csr_matrix.h"
#include "spmv/spmv.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

int main() {
    // 1. 创建 CSR 矩阵
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);

    // 2. 准备向量
    std::vector<float> x = {1, 1, 1};
    CudaBuffer<float> d_x(3), d_y(3);
    d_x.copyFromHost(x.data(), 3);

    // 3. 执行 SpMV
    SpMVConfig config = spmv_auto_config(csr);
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(), &config, 3);

    // 4. 获取结果
    std::vector<float> y(3);
    d_y.copyToHost(y.data(), 3);
    
    csr_destroy(csr);
    return 0;
}
```

---

## 执行上下文复用

```cpp
SpMVExecutionContext context;
SpMVConfig config;
config.use_texture = true;

// 多次执行，纹理对象复用
for (int iter = 0; iter < 100; iter++) {
    SpMVResult result = spmv_csr(csr, d_x.get(), d_y.get(),
                                 &config, n, &context);
}
```

---

## PageRank 应用

```cpp
#include "spmv/pagerank.h"

// 创建归一化邻接矩阵
csr_to_gpu(adj);

PageRankConfig config;
config.damping_factor = 0.85f;
config.tolerance = 1e-6f;

PageRankResult result = pagerank(adj, &config);

if (result.converged) {
    std::vector<TopKNode> top10(10);
    pagerank_top_k(&result, adj->num_rows, 10, top10.data());
}

pagerank_free(&result);
```
