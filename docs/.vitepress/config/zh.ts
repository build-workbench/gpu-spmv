import { defineConfig } from 'vitepress'

export const zh = defineConfig({
  lang: 'zh-CN',
  description: '基于 CUDA 的高性能稀疏矩阵向量乘法库',

  themeConfig: {
    nav: [
      { text: '快速开始', link: '/zh/quickstart' },
      { text: 'API 参考', link: '/zh/api/spmv' },
      { text: '架构设计', link: '/zh/architecture/overview' },
      { text: '性能', link: '/zh/performance/benchmarks' },
      { text: '学术参考', link: '/zh/references' }
    ],

    sidebar: {
      '/zh/': [
        {
          text: '开始',
          items: [
            { text: '介绍', link: '/zh/' },
            { text: '快速开始', link: '/zh/quickstart' },
            { text: '示例代码', link: '/zh/examples/basic-spmv' }
          ]
        },
        {
          text: 'API 参考',
          collapsed: false,
          items: [
            { text: 'SpMV 计算', link: '/zh/api/spmv' },
            { text: 'CSR 矩阵', link: '/zh/api/csr-matrix' },
            { text: 'ELL 矩阵', link: '/zh/api/ell-matrix' },
            { text: 'PageRank', link: '/zh/api/pagerank' }
          ]
        },
        {
          text: '架构设计',
          collapsed: false,
          items: [
            { text: '系统概览', link: '/zh/architecture/overview' },
            { text: 'Kernel 选择策略', link: '/zh/architecture/kernel-selection' },
            { text: '内存布局', link: '/zh/architecture/memory-layout' },
            { text: 'Spec-Driven 开发', link: '/zh/architecture/spec-driven' }
          ]
        },
        {
          text: '性能优化',
          collapsed: false,
          items: [
            { text: '基准测试', link: '/zh/performance/benchmarks' },
            { text: '优化指南', link: '/zh/performance/optimization-guide' }
          ]
        },
        {
          text: '社区',
          items: [
            { text: '学术参考', link: '/zh/references' },
            { text: '贡献指南', link: '/zh/contributing' },
            { text: '更新日志', link: '/zh/changelog' }
          ]
        }
      ]
    },

    editLink: {
      pattern: 'https://github.com/LessUp/gpu-spmv/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    lastUpdated: {
      text: '最后更新于'
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})
