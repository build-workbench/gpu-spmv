# Performance Methodology

## Read the method before the numbers

Benchmark numbers are not persuasive on their own. This page explains **under what conditions the measurements make sense, how they should be read, and which conclusions are safe to draw**.

## Measurement Assumptions

- GPU: NVIDIA RTX 3090 (Ampere)
- Peak bandwidth: 936 GB/s
- Primary metrics: time, bandwidth, utilization, variance
- Main comparison: kernel choice across different sparsity patterns

## Recommended Reading Order

1. **Look for trends, not only peaks**: does the implementation stay near 70%+ utilization consistently?
2. **Read matrix pattern together with kernel choice**: regular and highly skewed matrices should not be judged the same way.
3. **Check whether the selector is explainable**: does the chosen kernel match the matrix statistics?
4. **Look at variance**: a high average with unstable spread is weaker evidence.
