# ELL Format Storage

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: ELL Matrix Storage
**Name**: ell-matrix-storage
**Text**: Support ELL (ELLPACK) format for sparse matrices with uniform row lengths, optimized for GPU coalesced memory access.

### Scenario: Dense-to-ELL Conversion
**WHEN** converting a dense matrix to ELL format
**THEN** all non-zero elements and their positions should be preserved accurately

### Scenario: Padding Correctness
**WHEN** a row has fewer non-zero elements than max_nnz_per_row
**THEN** should pad with zeros and invalid column indices (-1)

### Scenario: Column-Major Layout
**WHEN** accessing ELL matrix data
**THEN** data should be stored in column-major order for GPU coalesced access

### Scenario: Serialization Round Trip
**WHEN** serializing ELL to binary file and deserializing
**THEN** the deserialized ELL should match the original exactly

### Scenario: Storage Structure
**WHEN** creating an ELL matrix
**THEN** it should use two 2D arrays: values and column_indices, with each row padded to max_nnz_per_row

---

## Data Structure

```cpp
struct ELLMatrix {
    int num_rows;           // Number of rows
    int num_cols;           // Number of columns
    int max_nnz_per_row;    // Maximum non-zero elements per row
    int nnz;                // Actual total non-zero elements

    // Column-major storage for coalesced access
    float* values;          // Values array [num_rows * max_nnz_per_row]
    int* col_indices;       // Column indices [-1 indicates padding]

    float* d_values;        // GPU device pointers
    int* d_col_indices;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

## Column-Major Storage Explanation

```
Sparse Matrix (max_nnz_per_row = 2):
| 1 0 2 |     Row 0: [1, 2] columns [0, 2]
| 3 4 0 | =>  Row 1: [3, 4] columns [0, 1]
| 5 0 0 |     Row 2: [5, -] columns [0, -]

Column-major storage:
values:     [1, 3, 5, 2, 4, 0]     // Stored by column
col_indices: [0, 0, 0, 2, 1, -1]   // -1 indicates padding

GPU access: Thread i accesses values[k*num_rows + i], contiguous addresses!
```

## Invariants

- Padding elements use `col_indices == -1`
- Storage is column-major: `values[k * num_rows + i]` for row i, slot k
- `max_nnz_per_row >= actual max nnz in any row`

## Test Properties

| Property | Description |
|----------|-------------|
| P4 | ELL Dense-to-Sparse Round Trip |
| P5 | ELL Padding Correctness |
| P6 | ELL Column-Major Layout |
| P7 | ELL Serialization Round Trip |

## See Also

- [Public API](../public-api/spec.md) - API functions for ELL operations
- [CSR Format](../csr-format/spec.md) - Alternative sparse matrix format
