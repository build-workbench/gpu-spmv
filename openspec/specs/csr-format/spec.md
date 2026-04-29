# CSR Format Storage

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: CSR Matrix Storage
**Name**: csr-matrix-storage
**Text**: Support CSR (Compressed Sparse Row) format for efficient sparse matrix storage with minimal memory footprint.

### Scenario: Dense-to-CSR Conversion
**WHEN** converting a dense matrix to CSR format
**THEN** all non-zero elements and their positions should be preserved accurately

### Scenario: Element Lookup
**WHEN** querying element at position (i, j) using csr_get_element
**THEN** the correct value (non-zero or zero) should be returned

### Scenario: Serialization Round Trip
**WHEN** serializing CSR to binary file and deserializing
**THEN** the deserialized CSR should match the original exactly

### Scenario: Large Matrix Support
**WHEN** storing a matrix with up to 10 million non-zero elements
**THEN** the operation should complete successfully

### Scenario: Storage Structure
**WHEN** creating a CSR matrix
**THEN** it should use three arrays: values (non-zero element values), column_indices (column indices), row_pointers (row pointers)

---

## Data Structure

```cpp
struct CSRMatrix {
    int num_rows;           // Number of rows
    int num_cols;           // Number of columns
    int nnz;                // Total non-zero elements

    float* values;          // Non-zero values array [nnz]
    int* col_indices;       // Column indices array [nnz]
    int* row_ptrs;          // Row pointers array [num_rows + 1]

    // GPU device pointers
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    bool owns_host_memory;
    bool owns_device_memory;
};
```

## Invariants

- `row_ptrs[0] == 0`
- `row_ptrs[num_rows] == nnz`
- `row_ptrs[i] <= row_ptrs[i+1]` for all i
- All `col_indices[j]` must be in range `[0, num_cols)`

## Test Properties

| Property | Description |
|----------|-------------|
| P1 | CSR Dense-to-Sparse Round Trip |
| P2 | CSR Element Lookup Correctness |
| P3 | CSR Serialization Round Trip |

## See Also

- [Public API](../public-api/spec.md) - API functions for CSR operations
- [RFC 0001](/tmp/specs-backup/rfc/0001-core-architecture.md) - Original architecture design
