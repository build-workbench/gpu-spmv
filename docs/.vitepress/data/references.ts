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
      meta: 'Vendor baseline and API reference for sparse GPU primitives',
      url: 'https://developer.nvidia.com/cusparse'
    },
    {
      key: 'ginkgo',
      title: 'Ginkgo',
      meta: 'Production-quality sparse linear algebra library with strong docs',
      url: 'https://github.com/ginkgo-project/ginkgo'
    },
    {
      key: 'moderngpu',
      title: 'ModernGPU',
      meta: 'Useful for understanding scan / merge / partitioning techniques on GPU',
      url: 'https://github.com/moderngpu/moderngpu'
    },
    {
      key: 'suitesparse',
      title: 'SuiteSparse Matrix Collection',
      meta: 'Representative real-world sparse matrices for benchmark reasoning',
      url: 'https://github.com/DrTimothyAldenDavis/SuiteSparse'
    }
  ]
}
