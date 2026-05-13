# PageRank

PageRank algorithm implementation using SpMV.

## Configuration

```cpp
struct PageRankConfig {
    float damping_factor;  // Damping factor (default: 0.85)
    float tolerance;       // Convergence threshold (default: 1e-6)
    int max_iterations;    // Max iterations (default: 100)
};
```

## Result

```cpp
struct PageRankResult {
    float* ranks;          // PageRank scores [num_nodes]
    int iterations;        // Actual iterations performed
    float final_residual;  // Final residual
    bool converged;        // Whether converged
    int error_code;        // 0 = success
};

struct TopKNode {
    int node_id;  // Node ID
    float rank;   // PageRank score
};
```

## Core Functions

### Compute PageRank

```cpp
PageRankResult pagerank(const CSRMatrix* adj_matrix,
                        const PageRankConfig* config = nullptr);
```

### Get Top-K Nodes

```cpp
void pagerank_top_k(const PageRankResult* result, int num_nodes,
                    int k, TopKNode* top_k);
```

### Free Result

```cpp
void pagerank_free(PageRankResult* result);
```

## Algorithm

The PageRank algorithm computes the stationary distribution of a random walk:

$$r_{k+1} = d \cdot A \cdot r_k + \frac{1-d}{n}$$

Where:
- $r_k$ is the PageRank vector at iteration $k$
- $A$ is the normalized adjacency matrix
- $d$ is the damping factor (typically 0.85)
- $n$ is the number of nodes

## Example

```cpp
#include <spmv/pagerank.h>

int main() {
    // Create adjacency matrix for a graph
    CSRMatrix* adj = create_graph_adjacency();
    csr_to_gpu(adj);

    // Configure PageRank
    PageRankConfig config = {
        .damping_factor = 0.85f,
        .tolerance = 1e-6f,
        .max_iterations = 100
    };

    // Compute PageRank
    PageRankResult result = pagerank(adj, &config);

    if (result.converged) {
        printf("Converged in %d iterations\n", result.iterations);

        // Get top 10 nodes
        TopKNode top_k[10];
        pagerank_top_k(&result, adj->num_rows, 10, top_k);

        printf("Top 10 nodes:\n");
        for (int i = 0; i < 10; i++) {
            printf("  Node %d: %.6f\n", top_k[i].node_id, top_k[i].rank);
        }
    }

    pagerank_free(&result);
    csr_destroy(adj);
    return 0;
}
```

## Performance

PageRank is essentially repeated SpMV, so kernel selection applies:

| Graph Type | Nodes | Edges | Iterations | Time |
|:-----------|:-----:|:-----:|:----------:|-----:|
| Web graph | 1M | 10M | 15 | 3.5s |
| Social network | 500K | 5M | 12 | 1.8s |
| Citation network | 100K | 1M | 8 | 0.4s |
