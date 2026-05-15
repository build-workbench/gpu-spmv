export type SiteMetric = {
  label: string
  value: string
  description?: string
}

export const siteData = {
  repo: 'https://github.com/AICL-Lab/gpu-spmv',
  zh: {
    heroTitle: 'GPU SpMV：技术白皮书与架构展示站',
    heroLead: '把 CUDA 稀疏矩阵向量乘法项目打造成可读、可证、可展示的工程作品。',
    metrics: [
      { label: 'Bandwidth Utilization', value: '70%+' },
      { label: 'Adaptive Kernels', value: '4' },
      { label: 'Sparse Formats', value: 'CSR + ELL' },
      { label: 'Property Tests', value: '100+' }
    ] satisfies SiteMetric[]
  },
  en: {
    heroTitle: 'GPU SpMV: Technical Whitepaper and Architecture Showcase',
    heroLead: 'Present the CUDA sparse matrix-vector multiplication project as a serious engineering artifact.',
    metrics: [
      { label: 'Bandwidth Utilization', value: '70%+' },
      { label: 'Adaptive Kernels', value: '4' },
      { label: 'Sparse Formats', value: 'CSR + ELL' },
      { label: 'Property Tests', value: '100+' }
    ] satisfies SiteMetric[]
  }
}
