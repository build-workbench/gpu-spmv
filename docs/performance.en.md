-- -layout : default title : Performance nav_order : 5 permalink : / performance.en lang : en-- -

    <p align = "right"><a href = "performance">🇨🇳 简体中文</ a></ p>
#Performance
    { :.no_toc}

    Tuning strategies,
    benchmarks, and best practices.{: .fs-6 .fw-300
}

##Table of Contents {: .no_toc .text-delta
}

1. TOC {:toc
}

-- -

        ##Performance Overview

            GPU SpMV achieves high performance through intelligent kernel scheduling :

    -**Target bandwidth utilization** : > 60 % -**Measured peak** : 70 - 75 % (RTX 3090) -
                                              **Scalability** : Linear growth with matrix size

                                                                -- -

                                              ##Benchmark Results

                                              ## #NVIDIA RTX 3090(Ampere, 936 GB / s)

    | Matrix Size | Non - zeros | Kernel | Bandwidth | | : -- -- -- -- -- - : | : -- -- -- -- - : |
    : -- -- -- - | : -- -- -- -- - : | | 10K × 10K | 500K | Vector CSR | **70.2 % ** | |
                                     100K × 100K | 5M | Merge Path | **71.5 % ** | | 1M × 1M |
                                     50M | Merge Path | **70.8 % ** |

                                     ## #GPU Architecture Comparison

                                     |
                                     GPU Architecture | Theoretical BW | Measured Util | |
    : -- -- -- -- -- -- -- -- - | : -- -- -- -- -- -- -- : |
    : -- -- -- -- -- -- - : | | Volta(V100) | 900 GB / s | ~65 % | | Turing(RTX 2080) | 448 GB / s |
                            ~68 % | | Ampere(RTX 3090) | 936 GB / s | ~70 % |
                            | Ada Lovelace(RTX 4090) | 1008 GB / s | ~72 % |

                            -- -

                                    ##Kernel Selection Strategy

``` Matrix Analysis
       │
       ├── avg nnz / row < 4 ──→ Scalar CSR(1 thread / row)
       │
       ├── avg nnz / row ≥ 4
       │       │
       │       ├── skewness < 10 ──→ Vector CSR(warp / row)
       │       │
       │       └── skewness ≥ 10 ──→ Merge Path(balanced)
       │
       └── ELL format ─────────→ ELL Kernel(coalesced)
```

                                ## #Kernel Types

                            |
                            Kernel | Best For | Bandwidth | | : -- -- -- - | : -- -- -- -- - |
    : -- -- -- -- - : | | Scalar CSR | Very sparse | ~40 - 50 % | | Vector CSR | Uniform dist |
                      ~65 - 75 % | | Merge Path | High skewness | ~70 - 80 % |
                      | ELL | Uniform rows | ~80 - 90 % |

                      -- -

                          ##Optimization Guide

                          ## #1. Use Auto
                          - Config(Recommended)

```cpp
                          // Library auto-selects optimal kernel
                          SpMVConfig config = spmv_auto_config(csr);
SpMVResult result = spmv_csr(csr, d_x, d_y, &config, n);
```

    ## #2. Format Selection

```cpp
    // Convert to ELL when row lengths are uniform
    if (row_length_variance < 0.2) {
    ELLMatrix* ell = ell_create(rows, cols, max_nnz_per_row);
    ell_from_csr(ell, csr);
    ell_to_gpu(ell);
    // ELL often performs better
}
```

    ## #3. Memory Optimization

```cpp
        // ✅ Use RAII for automatic management
        CudaBuffer<float>
            buffer(n);

// ❌ Avoid manual allocation
float* ptr;
cudaMalloc(&ptr, n * sizeof(float));
// Easy to forget cudaFree
```

            -- -

            ##Profiling Tools

```bash
#Nsight Systems - overall analysis
                nsys profile./
            spmv_benchmark

#Nsight Compute - detailed kernel analysis
                ncu-- kernel -
        name spmv./
            spmv_benchmark

#Built - in benchmark
                ./
            build -
        release / spmv_benchmark
```

            -- -

            ##Troubleshooting

            ## #Performance Not as Expected
    ?

    Checklist
    :

    1. ✅ Using `spmv_auto_config()` 2. ✅ Matrix transferred to
        GPU(`csr_to_gpu`) 3. ✅ Input vectors on GPU 4. ✅ Matrix large enough(> 10K non - zeros)

            -- -

        <div class = "text-center text-small" style = "margin-top: 3rem;"><p> Full benchmark data
            in<a href = "https://github.com/LessUp/gpu-spmv/tree/master/benchmarks"> benchmarks
            / </ a> directory</ p></ div>
