# ELL Matrix

ELL (ELLPACK) matrix data structure and operations.

## Data Structure

```cpp
struct ELLMatrix {
    int num_rows;         // Number of rows
    int num_cols;         // Number of columns
    int max_nnz_per_row;  // Max non-zeros per row (determines padding)
    int nnz;              // Actual total non-zeros

    // Column-major storage: values[k * num_rows + row]
    float* values;     // Values [num_rows * max_nnz_per_row]
    int* col_indices;  // Column indices, -1 indicates padding

    // GPU device pointers
    float* d_values;
    int* d_col_indices;

    // Memory ownership flags
    bool owns_host_memory;
    bool owns_device_memory;
};
```

## Core Functions

### Creation & Destruction

```cpp
ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row);
void ell_destroy(ELLMatrix* mat);
```

### Data Conversion

```cpp
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
int ell_to_dense(const ELLMatrix* ell, float* dense);
```

### GPU Data Transfer

```cpp
int ell_to_gpu(ELLMatrix* mat);
int ell_from_gpu(ELLMatrix* mat);
void ell_free_gpu(ELLMatrix* mat);
```

### Element Access

```cpp
float ell_get_element(const ELLMatrix* mat, int row, int col);
```

### Serialization

```cpp
int ell_serialize(const ELLMatrix* mat, const char* filename);
int ell_deserialize(ELLMatrix* mat, const char* filename);
```

### Validation

```cpp
bool ell_validate(const ELLMatrix* mat);
```

## Memory Layout

```
Original Matrix:     ELL Storage (column-major):
┌─────┬─────┐        values:     [ 1, 3, 5,   ← column 0
│ 1 0 2 │                      2, 4, 0 ]  ← column 1
│ 0 3 4 │   =>   col_indices: [ 0, 1, 3,   ← column 0
│ 0 0 5 │                      2, 2, - ]  ← column 1 (-1 = padding)
└─────┴─────┘        
                     Each row padded to max length
```

**Characteristics**:
- Column-major storage for GPU coalesced access
- No row pointers needed (reduces memory access)
- Aligned padding for SIMD execution
- Best performance for uniform row lengths

## When to Use ELL

| Condition | Recommendation |
|:----------|:---------------|
| Row lengths uniform (skewness < 3) | Use ELL |
| Row lengths vary significantly | Use CSR |
| Need maximum GPU performance | Convert to ELL |

## Example

```cpp
#include <spmv/spmv.h>

int main() {
    // Create CSR matrix first
    CSRMatrix* csr = csr_create(1000, 1000, 20000);
    // ... fill data ...

    // Check if ELL conversion is worthwhile
    CSRStats stats = csr_compute_stats(csr);
    if (stats.skewness < 3.0f) {
        // Convert to ELL
        ELLMatrix* ell = ell_create(csr->num_rows, csr->num_cols,
                                    stats.max_nnz_per_row);
        ell_from_csr(ell, csr);
        ell_to_gpu(ell);

        // Execute ELL SpMV
        SpMVResult result = spmv_ell(ell, d_x, d_y, nullptr);

        ell_destroy(ell);
    }

    csr_destroy(csr);
    return 0;
}
```
