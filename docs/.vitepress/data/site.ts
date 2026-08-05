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
      { label: '带宽利用率', value: '70%+' },
      { label: '自适应内核', value: '4' },
      { label: '稀疏格式', value: 'CSR + ELL' },
      { label: '属性测试', value: '100+' }
    ] satisfies SiteMetric[]
  }
}
