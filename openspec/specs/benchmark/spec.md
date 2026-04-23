# Performance Benchmarking

> **Version**: v1.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2025-04-16

## Requirement: Benchmark Suite
**Name**: benchmark-suite
**Text**: Provide comprehensive benchmarking tools to measure and compare SpMV performance across different implementations.

### Scenario: Metrics Collection
**WHEN** running benchmark for a SpMV kernel
**THEN** should measure execution time, GFLOPS, and bandwidth utilization

### Scenario: Statistical Reporting
**WHEN** running multiple benchmark iterations
**THEN** should report avg, min, max, and stddev across all runs

### Scenario: CPU Comparison
**WHEN** compare_cpu is enabled
**THEN** should run GPU implementation against CPU baseline and report comparison

### Scenario: JSON Export
**WHEN** export_json is enabled
**THEN** should generate JSON-formatted performance report for analysis

### Scenario: Standard Test Sets
**WHEN** running benchmark with external matrix files
**THEN** should support standard sparse matrix test sets (e.g., SuiteSparse collection)

---

## Data Structures

```cpp
struct BenchmarkConfig {
    int iterations;         // Number of benchmark iterations
    bool compare_cpu;       // Whether to run CPU baseline
    bool export_json;       // Whether to export JSON report
    const char* json_path;  // Path for JSON output
};

struct BenchmarkResult {
    float avg_time_ms;      // Average execution time
    float min_time_ms;      // Minimum time
    float max_time_ms;      // Maximum time
    float stddev_ms;        // Standard deviation
    float gflops;           // GFLOPS achieved
    float bandwidth_gb_s;   // Bandwidth utilization
    float cpu_time_ms;      // CPU baseline time (if enabled)
};
```

## Metrics Formulas

| Metric | Formula |
|--------|---------|
| GFLOPS | `2 × nnz / (time × 10⁹)` |
| Bandwidth | `bytes_accessed / elapsed_time` |
| Bytes Accessed | `(nnz × sizeof(float) × 2) + (nnz × sizeof(int)) + ...` |

## Test Properties

| Property | Description |
|----------|-------------|
| P13 | Benchmark Metrics Completeness |
| P14 | Benchmark JSON Round Trip |

## See Also

- [SpMV Kernels](../spmv-kernels/spec.md) - Kernel implementations
- [Public API](../public-api/spec.md) - Benchmark API functions
