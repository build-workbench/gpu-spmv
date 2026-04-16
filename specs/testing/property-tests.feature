# Property-Based Testing Specifications

> **Version**: v1.0.0  
> **Status**: ✅ Implemented  
> **Last Updated**: 2025-04-16

---

## Overview

This document defines the property-based test specifications for the GPU SpMV library. All property tests run a minimum of 100 iterations with randomly generated matrices.

---

## Test Framework

| Component | Technology |
|-----------|------------|
| Unit Testing | Google Test (GTest) |
| Property-Based Testing | Google Test + Random Generation |
| Performance Testing | CUDA Events for timing |
| Test Coverage Target | Core functionality > 80% |

---

## Property Tests

### Property 1: CSR Dense-to-Sparse Round Trip

**Validates**: Requirements 1.2

**Description**: Converting a dense matrix to CSR format and back should produce an identical dense matrix.

```gherkin
Given a random dense matrix with dimensions R x C
And a sparsity density between 0.1% and 30%
When converting dense matrix to CSR format
And converting CSR back to dense matrix
Then the resulting dense matrix should match the original exactly
```

**Test Implementation**:
```cpp
TEST(SpMVPropertyTest, CSRDenseToSparseRoundTrip) {
    for (int iter = 0; iter < 100; iter++) {
        auto matrix = generate_random_dense_matrix();
        CSRMatrix* csr = csr_from_dense(matrix);
        float* reconstructed = csr_to_dense(csr);
        
        EXPECT_TRUE(matrices_equal(matrix, reconstructed));
        
        csr_destroy(csr);
        free(reconstructed);
    }
}
```

---

### Property 2: CSR Element Lookup Correctness

**Validates**: Requirements 1.3

**Description**: Querying any element (i, j) from CSR should return the correct value.

```gherkin
Given a CSR matrix created from a dense matrix
When querying element (i, j) using csr_get_element
Then the returned value should match the original dense matrix value at (i, j)
```

**Test Implementation**:
```cpp
TEST(SpMVPropertyTest, CSRElementLookupCorrectness) {
    for (int iter = 0; iter < 100; iter++) {
        auto dense = generate_random_dense_matrix();
        CSRMatrix* csr = csr_from_dense(dense);
        
        for (int i = 0; i < csr->num_rows; i++) {
            for (int j = 0; j < csr->num_cols; j++) {
                EXPECT_FLOAT_EQ(csr_get_element(csr, i, j), dense[i][j]);
            }
        }
        
        csr_destroy(csr);
    }
}
```

---

### Property 3: CSR Serialization Round Trip

**Validates**: Requirements 1.5

**Description**: Serializing CSR to disk and deserializing should produce an equivalent CSR structure.

```gherkin
Given a CSR matrix with valid data
When serializing to binary file
And deserializing from that file
Then the deserialized CSR should match the original exactly
```

---

### Property 4: ELL Dense-to-Sparse Round Trip

**Validates**: Requirements 2.2

**Description**: Converting a dense matrix to ELL format and back should produce an identical dense matrix.

```gherkin
Given a random dense matrix with dimensions R x C
And a sparsity density between 0.1% and 30%
When converting dense matrix to ELL format
And converting ELL back to dense matrix
Then the resulting dense matrix should match the original exactly
```

---

### Property 5: ELL Padding Correctness

**Validates**: Requirements 2.3

**Description**: ELL format should correctly pad rows with fewer non-zero elements than max_nnz_per_row.

```gherkin
Given an ELL matrix created from a dense matrix
When examining each row's padding elements
Then padding elements should have column_index == -1
And padding elements should have value == 0
```

---

### Property 6: ELL Column-Major Layout

**Validates**: Requirements 2.4

**Description**: ELL format should store data in column-major order for GPU coalesced access.

```gherkin
Given an ELL matrix with known data
When accessing values using column-major indexing: values[k * num_rows + i]
Then the accessed value should match the expected value for row i, slot k
```

---

### Property 7: ELL Serialization Round Trip

**Validates**: Requirements 2.5

**Description**: Serializing ELL to disk and deserializing should produce an equivalent ELL structure.

```gherkin
Given an ELL matrix with valid data
When serializing to binary file
And deserializing from that file
Then the deserialized ELL should match the original exactly
```

---

### Property 8: SpMV CSR Correctness

**Validates**: Requirements 3.1, 3.3

**Description**: GPU SpMV computation on CSR format should match CPU reference implementation within tolerance.

```gherkin
Given a CSR matrix and input vector x
When executing SpMV on GPU
And executing SpMV on CPU reference
Then the relative error |y_gpu - y_cpu| / |y_cpu| should be < 1e-6 for all elements
```

**Test Implementation**:
```cpp
TEST(SpMVPropertyTest, SpMVCSRCorrectness) {
    for (int iter = 0; iter < 100; iter++) {
        auto matrix = generate_random_sparse_matrix();
        auto x = generate_random_vector(matrix->num_cols);
        
        // GPU computation
        SpMVResult gpu_result = spmv_csr(matrix, d_x, d_y, &config);
        
        // CPU reference
        spmv_cpu_csr(matrix, x.data(), y_cpu.data());
        
        // Verify relative error
        for (int i = 0; i < matrix->num_rows; i++) {
            if (y_cpu[i] != 0) {
                EXPECT_LT(abs(y_gpu[i] - y_cpu[i]) / abs(y_cpu[i]), 1e-6);
            }
        }
    }
}
```

---

### Property 9: SpMV ELL Correctness

**Validates**: Requirements 3.2, 3.3

**Description**: GPU SpMV computation on ELL format should match CPU reference implementation within tolerance.

```gherkin
Given an ELL matrix and input vector x
When executing SpMV on GPU
And executing SpMV on CPU reference
Then the relative error |y_gpu - y_cpu| / |y_cpu| should be < 1e-6 for all elements
```

---

### Property 10: SpMV Dimension Validation

**Validates**: Requirements 3.5, 8.5

**Description**: SpMV should reject invalid dimension combinations with appropriate error codes.

```gherkin
Given a CSR matrix with N columns
And an input vector with size M where M != N
When executing SpMV
Then the function should return INVALID_DIMENSION error code
```

**Test Cases**:
- Input vector smaller than matrix columns
- Input vector larger than matrix columns
- Output vector smaller than matrix rows
- Output vector larger than matrix rows
- Empty matrix (0 rows or 0 columns)

---

### Property 11: Kernel Selector Validity

**Validates**: Requirements 4.5

**Description**: Automatic kernel selector should choose appropriate kernel based on matrix characteristics.

```gherkin
Given a matrix with avg_nnz_per_row < 4
When calling spmv_auto_config
Then the selected kernel should be SCALAR_CSR

Given a matrix with avg_nnz_per_row >= 4 AND skewness < 10
When calling spmv_auto_config
Then the selected kernel should be VECTOR_CSR

Given a matrix with avg_nnz_per_row >= 4 AND skewness >= 10
When calling spmv_auto_config
Then the selected kernel should be MERGE_PATH
```

---

### Property 12: Bandwidth Metrics Validity

**Validates**: Requirements 5.5

**Description**: Bandwidth metrics should be computed correctly and fall within realistic bounds.

```gherkin
Given a SpMV operation with known data access size
And measured execution time
When computing bandwidth
Then bandwidth should equal bytes_accessed / elapsed_time
And bandwidth should not exceed GPU theoretical peak
```

---

### Property 13: Benchmark Metrics Completeness

**Validates**: Requirements 6.1, 6.3

**Description**: Benchmark suite should report all required statistical metrics.

```gherkin
Given a benchmark run with multiple iterations
When analyzing benchmark results
Then avg_time_ms should be the arithmetic mean of all iterations
And min_time_ms should be the minimum observed time
And max_time_ms should be the maximum observed time
And stddev_ms should be the standard deviation
And gflops should be computed as 2 * nnz / (avg_time * 10^9)
And bandwidth_gb_s should be computed from bytes accessed
```

---

### Property 14: Benchmark JSON Round Trip

**Validates**: Requirements 6.5

**Description**: Benchmark results serialized to JSON and parsed back should match original values.

```gherkin
Given benchmark results with all metrics populated
When serializing to JSON file
And parsing from that JSON file
Then all fields in parsed result should match original values exactly
```

---

### Property 15: PageRank Score Invariants

**Validates**: Requirements 7.1, 7.2

**Description**: PageRank scores should satisfy mathematical invariants.

```gherkin
Given a valid adjacency matrix
When computing PageRank
Then all PageRank scores should be non-negative
And the sum of all PageRank scores should equal 1.0 (within tolerance)
And if converged, final_residual should be < tolerance
```

**Test Implementation**:
```cpp
TEST(SpMVPropertyTest, PageRankScoreInvariants) {
    for (int iter = 0; iter < 100; iter++) {
        auto adj_matrix = generate_random_graph();
        PageRankResult result = pagerank(adj_matrix, &config);
        
        // Non-negative scores
        for (int i = 0; i < adj_matrix->num_rows; i++) {
            EXPECT_GE(result.ranks[i], 0.0f);
        }
        
        // Sum to 1.0
        float sum = 0.0f;
        for (int i = 0; i < adj_matrix->num_rows; i++) {
            sum += result.ranks[i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4);
        
        pagerank_free(&result);
    }
}
```

---

### Property 16: PageRank Top-K Ordering

**Validates**: Requirements 7.5

**Description**: Top-K nodes should be correctly sorted by PageRank score in descending order.

```gherkin
Given PageRank results and a value K
When extracting top-K nodes
Then the returned array should have exactly K elements
And scores should be in descending order: score[i] >= score[i+1]
And all returned nodes should be valid node indices
```

---

## Test Matrix Generator

### SparseMatrixGenerator

```cpp
struct SparseMatrixGenerator {
    int min_rows = 1, max_rows = 1000;
    int min_cols = 1, max_cols = 1000;
    float min_density = 0.001, max_density = 0.3;

    enum RowDistribution {
        UNIFORM,        // Each row has similar nnz count
        POWER_LAW,      // Power-law distribution (real-world graphs)
        EXTREME_SKEW    // Highly skewed row lengths
    };

    CSRMatrix* generate(RowDistribution dist = UNIFORM);
};
```

### Generator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_rows | 1 | Minimum matrix rows |
| max_rows | 1000 | Maximum matrix rows |
| min_cols | 1 | Minimum matrix columns |
| max_cols | 1000 | Maximum matrix columns |
| min_density | 0.001 | Minimum sparsity (0.1%) |
| max_density | 0.3 | Maximum sparsity (30%) |

---

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Empty matrix (0 rows or 0 cols) | Return empty result vector |
| All-zero rows | SpMV produces 0 for those rows |
| Single element matrix | Process normally |
| Extremely large matrix (exceeds GPU memory) | Return OUT_OF_MEMORY error |
| NaN/Inf input values | Propagate to output (IEEE 754 semantics) |
| Vector dimension mismatch | Return INVALID_DIMENSION error |

---

## Running Tests

```bash
# Run all tests
./build-release/spmv_tests

# Run specific property tests
./build-release/spmv_tests --gtest_filter="*Property*"

# Run with random matrix variations
./build-release/spmv_tests --gtest_repeat=10

# Run specific property
./build-release/spmv_tests --gtest_filter="SpMVPropertyTest.SpMVCSRCorrectness"
```

---

## Test Coverage Matrix

| Requirement | Properties | Test Status |
|-------------|------------|-------------|
| REQ-1 (CSR Storage) | P1, P2, P3 | ✅ Covered |
| REQ-2 (ELL Storage) | P4, P5, P6, P7 | ✅ Covered |
| REQ-3 (Basic SpMV) | P8, P9, P10 | ✅ Covered |
| REQ-4 (Load Balancing) | P11 | ✅ Covered |
| REQ-5 (Bandwidth) | P12 | ✅ Covered |
| REQ-6 (Benchmarking) | P13, P14 | ✅ Covered |
| REQ-7 (PageRank) | P15, P16 | ✅ Covered |
| REQ-8 (Error Handling) | All Properties | ✅ Covered |
