# Optimization Guide

How to achieve the best performance with GPU SpMV.

## 1. Choose the Right Kernel

### Automatic Selection

```cpp
SpMVConfig config = spmv_auto_config(csr);  // Recommended
```

### Manual Selection

| Scenario | Recommended Kernel | Reason |
|:---------|:-------------------|:-------|
| Very sparse (avg_nnz < 4) | Scalar CSR | Minimal overhead |
| Uniform distribution | Vector CSR | Efficient warp cooperation |
| Highly skewed | Merge Path | Perfect load balancing |
| Uniform row lengths | ELL Kernel | Fully coalesced access |

## 2. Use ELL Format

When matrix row lengths are uniform, ELL format provides best performance:

```cpp
CSRStats stats = csr_compute_stats(csr);

if (stats.skewness < 3.0f) {
    // Convert to ELL
    ELLMatrix* ell = ell_create(csr->num_rows, csr->num_cols,
                                stats.max_nnz_per_row);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);

    // Use ELL SpMV
    result = spmv_ell(ell, d_x, d_y, nullptr);
}
```

## 3. Reuse Execution Context

Reuse texture cache across iterations:

```cpp
SpMVExecutionContext ctx;  // Create once

for (int i = 0; i < iterations; i++) {
    // Texture object is reused, avoiding recreation
    result = spmv_csr(csr, d_x, d_y, &config, n, &ctx);
}

ctx.reset();  // Reset when done
```

## 4. Memory Layout Optimization

### Vector Alignment

```cpp
// Use CudaBuffer for alignment
CudaBuffer<float> d_x(N);
CudaBuffer<float> d_y(N);
```

### Batch Transfer

```cpp
// Transfer multiple vectors in one call
cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
// Instead of many small transfers
```

## 5. Tune Thresholds

Adjust selection thresholds for specific hardware:

```cpp
SpMVThresholds thresholds = spmv_get_thresholds();

// May need adjustment on newer GPUs
thresholds.avg_nnz_threshold = 3.0f;      // Lower Scalar CSR usage
thresholds.skewness_threshold = 15.0f;    // Higher Merge Path usage

spmv_set_thresholds(thresholds);
```

## 6. Performance Profiling

### Build a Small Measurement Loop

```cpp
SpMVExecutionContext ctx;
SpMVConfig config = spmv_auto_config(csr);

for (int i = 0; i < 5; ++i) {
    spmv_csr(csr, d_x, d_y, &config, csr->num_cols, &ctx);  // Warmup
}

SpMVResult result = spmv_csr(csr, d_x, d_y, &config, csr->num_cols, &ctx);
printf("Elapsed: %.3f ms\n", result.elapsed_ms);
printf("Bandwidth: %.1f GB/s\n", result.bandwidth_gb_s);
```

### Using Nsight

```bash
# Profile
nsys profile ./spmv_program

# Detailed analysis
ncu ./spmv_program
```

## Performance Checklist

- [ ] Use `spmv_auto_config()` for automatic selection
- [ ] Check if ELL format is suitable
- [ ] Reuse execution context in iterations
- [ ] Use `CudaBuffer` for memory management
- [ ] Verify bandwidth utilization > 60%

## References

- [Benchmarks](/en/performance/benchmarks)
- [Kernel Selection](/en/architecture/kernel-selection)