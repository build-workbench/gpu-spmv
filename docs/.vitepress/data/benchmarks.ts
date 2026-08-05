export const benchmarkData = {
  environment: [
    { label: 'GPU', value: 'RTX 3090' },
    { label: '理论带宽', value: '936 GB/s' },
    { label: 'CUDA', value: '12.0' },
    { label: 'CPU', value: 'Ryzen 9 5950X' }
  ],
  summary: [
    { label: '典型利用率', value: '70%+' },
    { label: '最优内核族', value: 'Merge Path' },
    { label: '最优规则模式', value: 'ELL' },
    { label: '选择器准确率', value: '100%' }
  ],
  scenarios: [
    { label: '极稀疏', value: 'Scalar CSR', description: 'avg_nnz_per_row < 4' },
    { label: '均匀行', value: 'Vector CSR', description: '低倾斜，warp 利用率好' },
    { label: '高倾斜', value: 'Merge Path', description: '行长不规则，负载均衡更好' },
    { label: 'ELL 友好', value: 'ELL Kernel', description: '行宽均匀，合并访存' }
  ]
}
