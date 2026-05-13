# PageRank

基于 SpMV 的 PageRank 算法实现。

## 配置

```cpp
struct PageRankConfig {
    float damping_factor;  // 阻尼因子（默认 0.85）
    float tolerance;       // 收敛阈值（默认 1e-6）
    int max_iterations;    // 最大迭代次数（默认 100）
};
```

## 结果

```cpp
struct PageRankResult {
    float* ranks;          // PageRank 分数 [num_nodes]
    int iterations;        // 实际迭代次数
    float final_residual;  // 最终残差
    bool converged;        // 是否收敛
    int error_code;        // 0 = 成功
};

struct TopKNode {
    int node_id;  // 节点 ID
    float rank;   // PageRank 分数
};
```

## 核心函数

### 计算 PageRank

```cpp
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config = nullptr);
```

### 获取 Top-K 节点

```cpp
void pagerank_top_k(const PageRankResult* result, int num_nodes,
                    int k, TopKNode* top_k);
```

### 释放结果

```cpp
void pagerank_free(PageRankResult* result);
```

## 算法

PageRank 算法计算随机游走的平稳分布：

$$r_{k+1} = d \cdot A \cdot r_k + \frac{1-d}{n}$$

其中：
- $r_k$ 是第 $k$ 次迭代的 PageRank 向量
- $A$ 是归一化的邻接矩阵
- $d$ 是阻尼因子（通常 0.85）
- $n$ 是节点数

## 示例

```cpp
#include <spmv/pagerank.h>

int main() {
    // 创建图的邻接矩阵
    CSRMatrix* adj = create_graph_adjacency();
    csr_to_gpu(adj);

    // 配置 PageRank
    PageRankConfig config = {
        .damping_factor = 0.85f,
        .tolerance = 1e-6f,
        .max_iterations = 100
    };

    // 计算 PageRank
    PageRankResult result = pagerank(adj, &config);

    if (result.converged) {
        printf("收敛于 %d 次迭代\n", result.iterations);

        // 获取前 10 个节点
        TopKNode top_k[10];
        pagerank_top_k(&result, adj->num_rows, 10, top_k);

        printf("Top 10 节点:\n");
        for (int i = 0; i < 10; i++) {
            printf("  节点 %d: %.6f\n", top_k[i].node_id, top_k[i].rank);
        }
    }

    pagerank_free(&result);
    csr_destroy(adj);
    return 0;
}
```

## 性能

PageRank 本质上是重复的 SpMV，因此 Kernel 选择同样适用：

| 图类型 | 节点数 | 边数 | 迭代次数 | 时间 |
|:-------|:------:|:----:|:--------:|-----:|
| 网页图 | 1M | 10M | 15 | 3.5s |
| 社交网络 | 500K | 5M | 12 | 1.8s |
| 引用网络 | 100K | 1M | 8 | 0.4s |
