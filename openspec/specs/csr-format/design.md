# CSR Format Design

## Context

CSR (Compressed Sparse Row) is one of the most commonly used sparse matrix storage formats, suitable for general-purpose sparse matrix operations. It provides memory-efficient storage for matrices with a large number of zero elements.

## Goals / Non-Goals

**Goals:**
- Efficient storage for sparse matrices with minimal memory overhead
- Support for GPU-accelerated computation
- Support for matrices with up to 10M+ non-zero elements
- Binary serialization for persistence

**Non-Goals:**
- Dynamic modification of matrix structure (add/remove elements)
- Support for non-numeric data types
- Column-wise access optimization

## Decisions

### D1: Data Structure Layout

CSR uses three arrays to represent a sparse matrix:

```
Sparse Matrix:              CSR Storage:
| 1 0 2 0 |                 values:      [1, 2, 3, 4, 5]
| 0 3 4 0 |      =>         col_indices: [0, 2, 1, 2, 3]
| 0 0 0 5 |                 row_ptrs:    [0, 2, 4, 5]
                            (Row 0: indices 0-1, 2 elements)
                            (Row 1: indices 2-3, 2 elements)
                            (Row 2: index 4,    1 element)
```

**Rationale**: This layout provides O(1) row access and O(log nnz_per_row) element lookup while minimizing memory usage.

### D2: Memory Management

Use `owns_host_memory` and `owns_device_memory` flags to track memory ownership:

```cpp
// Ownership flags prevent double-free
bool owns_host_memory;     // Free host memory on destroy?
bool owns_device_memory;   // Free device memory on destroy?
```

**Rationale**: Enables flexible memory management patterns including views and transfers.

### D3: GPU Memory Transfer

Explicit transfer functions rather than automatic synchronization:

```cpp
int csr_to_gpu(CSRMatrix* csr);    // Host -> Device
int csr_from_gpu(const CSRMatrix* csr);  // Device -> Host
```

**Rationale**: Gives developers control over transfer timing for performance optimization.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Poor column-wise access performance | Use ELL format or consider CSC for column-heavy workloads |
| Memory fragmentation for very large matrices | Pre-allocate with known nnz count |
| Modification requires full reconstruction | Document that CSR is immutable structure |

## Performance Considerations

- Row-wise access: O(nnz_per_row)
- Element lookup: O(nnz_per_row) worst case, O(log nnz_per_row) with binary search
- Memory overhead: O(nnz + num_rows) for values + indices + pointers
