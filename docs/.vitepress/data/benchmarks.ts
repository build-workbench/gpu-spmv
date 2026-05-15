export const benchmarkData = {
  environment: [
    { label: 'GPU', value: 'RTX 3090' },
    { label: 'Peak BW', value: '936 GB/s' },
    { label: 'CUDA', value: '12.0' },
    { label: 'CPU', value: 'Ryzen 9 5950X' }
  ],
  summary: [
    { label: 'Typical Utilization', value: '70%+' },
    { label: 'Best Kernel Family', value: 'Merge Path' },
    { label: 'Best Regular Pattern', value: 'ELL' },
    { label: 'Selector Accuracy', value: '100%' }
  ],
  scenarios: [
    { label: 'Very sparse', value: 'Scalar CSR', description: 'avg_nnz_per_row < 4' },
    { label: 'Uniform rows', value: 'Vector CSR', description: 'Low skewness, good warp utilization' },
    { label: 'High skew', value: 'Merge Path', description: 'Irregular row lengths with better balancing' },
    { label: 'ELL-friendly', value: 'ELL Kernel', description: 'Uniform row width, coalesced memory access' }
  ]
}
