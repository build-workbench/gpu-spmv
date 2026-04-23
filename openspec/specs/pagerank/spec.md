# PageRank Algorithm

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: PageRank Implementation
**Name**: pagerank-implementation
**Text**: Implement PageRank algorithm using iterative SpMV to demonstrate practical application of sparse matrix operations on graph data.

### Scenario: PageRank Computation
**WHEN** given an adjacency matrix and damping factor
**THEN** should compute PageRank scores using iterative SpMV

### Scenario: Convergence
**WHEN** L2 norm of ranking differences between iterations falls below tolerance (1e-6)
**THEN** should stop iteration and report convergence

### Scenario: Dangling Nodes
**WHEN** processing graphs with dangling nodes (no outgoing edges)
**THEN** should handle correctly by redistributing their rank mass

### Scenario: Large Graph Support
**WHEN** processing graphs with up to 1 million nodes
**THEN** should complete successfully

### Scenario: Top-K Output
**WHEN** requesting top-K nodes
**THEN** should output nodes sorted by ranking score in descending order

---

## Algorithm

**PageRank Iteration Formula:**
```
r_{k+1} = d × A × r_k + (1-d) / n
```

Where:
- `r_k` = PageRank vector at iteration k
- `A` = Column-normalized adjacency matrix
- `d` = Damping factor (typically 0.85)
- `n` = Number of nodes

**Convergence:**
```
||r_{k+1} - r_k||_2 < tolerance
```

## Data Structures

```cpp
struct PageRankConfig {
    float damping_factor = 0.85f;   // Damping factor (d)
    float tolerance = 1e-6f;        // Convergence threshold
    int max_iterations = 100;       // Maximum iterations
};

struct PageRankResult {
    float* ranks;           // PageRank scores [num_nodes]
    int iterations;         // Actual iterations performed
    float final_residual;   // Final L2 norm residual
    bool converged;         // Whether converged
    int error_code;         // Error code
};
```

## Test Properties

| Property | Description |
|----------|-------------|
| P15 | PageRank Score Invariants |
| P16 | PageRank Top-K Ordering |

## Invariants

- All PageRank scores must be non-negative
- Sum of all PageRank scores should equal 1.0 (within tolerance)
- If converged, `final_residual < tolerance`

## See Also

- [SpMV Kernels](../spmv-kernels/spec.md) - Core SpMV operation
- [CSR Format](../csr-format/spec.md) - Matrix storage
