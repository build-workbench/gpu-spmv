# Property-Based Testing Specifications

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

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

## Requirement: Property Testing
**Name**: property-testing
**Text**: Validate system properties through randomized testing with minimum 100 iterations.

### Scenario: Random Matrix Generation
**WHEN** generating random sparse matrices for testing
**THEN** should support various dimensions, densities, and row distributions

### Scenario: Property Verification
**WHEN** running property tests
**THEN** should verify mathematical invariants and correctness properties

---

## Property Tests

### Property 1: CSR Dense-to-Sparse Round Trip

**Validates**: Requirements 1.2

**WHEN** converting a dense matrix to CSR format and back
**THEN** the resulting dense matrix should match the original exactly

```cpp
TEST(SpMVPropertyTest, CSRDenseToSparseRoundTrip) {
    for (int iter = 0; iter < 100; iter++) {
        auto dense = generate_random_dense_matrix();
        CSRMatrix* csr = csr_from_dense(dense);
        float* reconstructed = csr_to_dense(csr);

        EXPECT_TRUE(matrices_equal(dense, reconstructed));

        csr_destroy(csr);
        free(reconstructed);
    }
}
```

---

### Property 2: CSR Element Lookup Correctness

**Validates**: Requirements 1.3

**WHEN** querying element (i, j) from CSR matrix
**THEN** the returned value should match the original dense matrix value

---

### Property 3: CSR Serialization Round Trip

**Validates**: Requirements 1.5

**WHEN** serializing CSR to binary file and deserializing
**THEN** the deserialized CSR should match the original exactly

---

### Property 4: ELL Dense-to-Sparse Round Trip

**Validates**: Requirements 2.2

**WHEN** converting a dense matrix to ELL format and back
**THEN** the resulting dense matrix should match the original exactly

---

### Property 5: ELL Padding Correctness

**Validates**: Requirements 2.3

**WHEN** examining padding elements in ELL matrix
**THEN** padding elements should have `column_index == -1` and `value == 0`

---

### Property 6: ELL Column-Major Layout

**Validates**: Requirements 2.4

**WHEN** accessing ELL matrix using column-major indexing
**THEN** the accessed value should match the expected value for row i, slot k

---

### Property 7: ELL Serialization Round Trip

**Validates**: Requirements 2.5

**WHEN** serializing ELL to binary file and deserializing
**THEN** the deserialized ELL should match the original exactly

---

### Property 8: SpMV CSR Correctness

**Validates**: Requirements 3.1, 3.3

**WHEN** executing SpMV on GPU with CSR format and comparing to CPU reference
**THEN** the relative error should be < 1e-6 for all elements

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

**WHEN** executing SpMV on GPU with ELL format and comparing to CPU reference
**THEN** the relative error should be < 1e-6 for all elements

---

### Property 10: SpMV Dimension Validation

**Validates**: Requirements 3.5, 8.5

**WHEN** executing SpMV with mismatched vector dimensions
**THEN** should return INVALID_DIMENSION error code

**Test Cases:**
- Input vector smaller than matrix columns
- Input vector larger than matrix columns
- Output vector smaller than matrix rows
- Output vector larger than matrix rows
- Empty matrix (0 rows or 0 columns)

---

### Property 11: Kernel Selector Validity

**Validates**: Requirements 4.5

**WHEN** calling spmv_auto_config with various matrix characteristics
**THEN** should select appropriate kernel:
- `avg_nnz_per_row < 4` → `SCALAR_CSR`
- `avg_nnz_per_row >= 4` AND `skewness < 10` → `VECTOR_CSR`
- `avg_nnz_per_row >= 4` AND `skewness >= 10` → `MERGE_PATH`

---

### Property 12: Bandwidth Metrics Validity

**Validates**: Requirements 5.5

**WHEN** computing bandwidth from SpMV operation
**THEN** bandwidth should equal `bytes_accessed / elapsed_time`
AND should not exceed GPU theoretical peak

---

### Property 13: Benchmark Metrics Completeness

**Validates**: Requirements 6.1, 6.3

**WHEN** running benchmark with multiple iterations
**THEN** should report:
- `avg_time_ms` = arithmetic mean
- `min_time_ms` = minimum observed
- `max_time_ms` = maximum observed
- `stddev_ms` = standard deviation
- `gflops` = `2 * nnz / (avg_time * 10^9)`
- `bandwidth_gb_s` = computed from bytes accessed

---

### Property 14: Benchmark JSON Round Trip

**Validates**: Requirements 6.5

**WHEN** serializing benchmark results to JSON and parsing back
**THEN** all fields should match original values exactly

---

### Property 15: PageRank Score Invariants

**Validates**: Requirements 7.1, 7.2

**WHEN** computing PageRank
**THEN** all scores should be non-negative
AND sum of all scores should equal 1.0 (within tolerance)
AND if converged, `final_residual < tolerance`

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

**WHEN** extracting top-K nodes by PageRank score
**THEN** returned array should have exactly K elements
AND scores should be in descending order
AND all returned nodes should be valid indices

---

## Test Matrix Generator

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

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| Empty matrix (0 rows or 0 cols) | Return empty result vector |
| All-zero rows | SpMV produces 0 for those rows |
| Single element matrix | Process normally |
| Extremely large matrix (exceeds GPU memory) | Return OUT_OF_MEMORY error |
| NaN/Inf input values | Propagate to output (IEEE 754 semantics) |
| Vector dimension mismatch | Return INVALID_DIMENSION error |

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
