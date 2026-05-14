# CSR Matrix

CSR (Compressed Sparse Row) matrix data structure and operations.

## Data Structure

```cpp
struct CSRMatrix {
    int num_rows;  // Number of rows
    int num_cols;  // Number of columns
    int nnz;       // Total non-zero elements

    float* values;     // Non-zero values [nnz]
    int* col_indices;  // Column indices [nnz]
    int* row_ptrs;     // Row pointers [num_rows + 1]

    void* internal;    // Opaque internal state (device memory management)
};
```

## Core Functions

### Creation & Destruction

```cpp
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* mat);
```

### Data Conversion

```cpp
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
int csr_to_dense(const CSRMatrix* csr, float* dense);
```

### GPU Data Transfer

```cpp
int csr_to_gpu(CSRMatrix* mat);
int csr_from_gpu(CSRMatrix* mat);
```

### Element Access

```cpp
float csr_get_element(const CSRMatrix* mat, int row, int col);
```

### Serialization

```cpp
int csr_serialize(const CSRMatrix* mat, const char* filename);
int csr_deserialize(CSRMatrix* mat, const char* filename);  // In-place style
```

### Statistics & Validation

```cpp
CSRStats csr_compute_stats(const CSRMatrix* mat);
bool csr_validate(const CSRMatrix* mat);
```

## CSRStats Structure

```cpp
struct CSRStats {
    float avg_nnz_per_row;  // Average non-zeros per row
    int max_nnz_per_row;    // Maximum non-zeros per row
    int min_nnz_per_row;    // Minimum non-zeros per row
    float skewness;         // Skewness: max / (min + 1)
};
```

## Memory Layout

```
Original Matrix:     CSR Storage:
┌─────┬─────┐        values:     [ 1, 2, 3, 4, 5 ]
│ 1 0 2 │        col_indices: [ 0, 2, 1, 2, 3 ]
│ 0 3 4 │   =>   row_ptrs:    [ 0, 2, 4, 5 ]
│ 0 0 5 │        
└─────┴─────┘        row_ptrs[i] indicates the starting position of row i
```

**Characteristics**:
- General-purpose format, memory efficient
- Suitable for irregular sparsity patterns
- Efficient row traversal

## Example

```cpp
#include <spmv/spmv.h>

int main() {
    // Create 3x3 CSR matrix with 5 non-zeros
    CSRMatrix* csr = csr_create(3, 3, 5);

    // Fill from dense array
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    csr_from_dense(csr, dense, 3, 3);

    // Compute statistics
    CSRStats stats = csr_compute_stats(csr);
    printf("Avg NNZ/row: %.2f\n", stats.avg_nnz_per_row);
    printf("Skewness: %.2f\n", stats.skewness);

    // Transfer to GPU
    csr_to_gpu(csr);

    // ... use for SpMV ...

    csr_destroy(csr);
    return 0;
}
```
