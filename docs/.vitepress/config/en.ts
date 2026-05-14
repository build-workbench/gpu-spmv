import { defineConfig } from 'vitepress'

export const en = defineConfig({
  lang: 'en-US',
  description: 'High-Performance CUDA Sparse Matrix-Vector Multiplication Library',

  themeConfig: {
    nav: [
      { text: 'Whitepaper', link: '/en/whitepaper/' },
      { text: 'Getting Started', link: '/en/quickstart' },
      { text: 'Architecture', link: '/en/architecture/overview' },
      { text: 'API Reference', link: '/en/api/spmv' },
      { text: 'Benchmarks', link: '/en/performance/benchmarks' }
    ],

    sidebar: {
      '/en/': [
        {
          text: '📄 Technical Whitepaper',
          collapsed: false,
          items: [
            { text: 'Executive Summary', link: '/en/whitepaper/' },
            { text: 'Design Philosophy', link: '/en/whitepaper/philosophy' },
            { text: 'Performance Analysis', link: '/en/whitepaper/performance' }
          ]
        },
        {
          text: '🚀 Getting Started',
          items: [
            { text: 'Introduction', link: '/en/' },
            { text: 'Quick Start', link: '/en/quickstart' },
            { text: 'Examples', link: '/en/examples/basic-spmv' }
          ]
        },
        {
          text: '🏗️ Architecture',
          collapsed: false,
          items: [
            { text: 'System Overview', link: '/en/architecture/overview' },
            { text: 'Kernel Selection', link: '/en/architecture/kernel-selection' },
            { text: 'Memory Layout', link: '/en/architecture/memory-layout' },
            { text: 'Spec-Driven Dev', link: '/en/architecture/spec-driven' }
          ]
        },
        {
          text: '⚡ Performance',
          collapsed: false,
          items: [
            { text: 'Benchmarks', link: '/en/performance/benchmarks' },
            { text: 'Optimization Guide', link: '/en/performance/optimization-guide' }
          ]
        },
        {
          text: '📚 API Reference',
          collapsed: true,
          items: [
            { text: 'SpMV Computation', link: '/en/api/spmv' },
            { text: 'CSR Matrix', link: '/en/api/csr-matrix' },
            { text: 'ELL Matrix', link: '/en/api/ell-matrix' },
            { text: 'PageRank', link: '/en/api/pagerank' }
          ]
        },
        {
          text: '🎓 Academic',
          items: [
            { text: 'References', link: '/en/references' },
            { text: 'Citation', link: '/en/citation' },
            { text: 'Contributing', link: '/en/contributing' },
            { text: 'Changelog', link: '/en/changelog' }
          ]
        }
      ]
    },

    editLink: {
      pattern: 'https://github.com/LessUp/gpu-spmv/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    outline: {
      label: 'On This Page',
      level: [2, 3]
    }
  }
})
