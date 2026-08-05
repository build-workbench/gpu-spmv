export const references = {
  papers: [
    {
      key: 'bell-garland-2009',
      title: 'Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors',
      meta: 'Nathan Bell, Michael Garland · SC 2009',
      url: 'https://doi.org/10.1145/1654059.1654121'
    },
    {
      key: 'merrill-garland-2016',
      title: 'Merge-based Parallel Sparse Matrix-Vector Multiplication',
      meta: 'Duane Merrill, Michael Garland · SC 2016',
      url: 'https://doi.org/10.1145/3016078.2851141'
    },
    {
      key: 'vazquez-ellrt-2011',
      title: 'Automatic Tuning of the Sparse Matrix Vector Product on GPUs Based on the ELL-R-T Format',
      meta: 'Fernando Vázquez et al. · Concurrency and Computation 2011',
      url: 'https://doi.org/10.1002/cpe.1761'
    }
  ],
  projects: [
    {
      key: 'cusparse',
      title: 'NVIDIA cuSPARSE',
      meta: '厂商基线与稀疏 GPU 原语 API 参考',
      url: 'https://developer.nvidia.com/cusparse'
    },
    {
      key: 'ginkgo',
      title: 'Ginkgo',
      meta: '生产级稀疏线性代数库，文档完善',
      url: 'https://github.com/ginkgo-project/ginkgo'
    },
    {
      key: 'moderngpu',
      title: 'ModernGPU',
      meta: '理解 GPU 上 scan / merge / partition 技术的参考',
      url: 'https://github.com/moderngpu/moderngpu'
    },
    {
      key: 'suitesparse',
      title: 'SuiteSparse Matrix Collection',
      meta: '代表性真实稀疏矩阵集合，用于基准推理',
      url: 'https://github.com/DrTimothyAldenDavis/SuiteSparse'
    }
  ]
}
