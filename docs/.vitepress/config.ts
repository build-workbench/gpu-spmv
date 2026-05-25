import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/'

export default withMermaid(
  defineConfig({
    base,
    title: 'GPU SpMV',
    description: 'High-Performance CUDA Sparse Matrix-Vector Multiplication',

    head: [
      ['meta', { name: 'theme-color', content: '#76B900' }],
      ['meta', { property: 'og:type', content: 'website' }],
      ['meta', { property: 'og:title', content: 'GPU SpMV' }],
      [
        'meta',
        {
          property: 'og:description',
          content: 'High-Performance CUDA Sparse Matrix-Vector Multiplication Library'
        }
      ],
      ['meta', { property: 'og:image', content: `${base}images/social/og-dark.svg` }],
      ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
      ['meta', { name: 'twitter:title', content: 'GPU SpMV' }],
      ['meta', { name: 'twitter:description', content: 'High-Performance CUDA Sparse Matrix-Vector Multiplication Library' }],
      ['meta', { name: 'twitter:image', content: `${base}images/social/og-dark.svg` }],
      ['link', { rel: 'icon', href: `${base}images/favicon.svg`, type: 'image/svg+xml' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
      ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
      [
        'link',
        {
          href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
          rel: 'stylesheet'
        }
      ]
    ],

    cleanUrls: true,
    lastUpdated: true,

    markdown: {
      lineNumbers: true
    },

    locales: {
      zh: {
        label: '简体中文',
        lang: 'zh-CN',
        link: '/zh/',
        description: '基于 CUDA 的高性能稀疏矩阵向量乘法库',
        themeConfig: {
          nav: [
            { text: '技术白皮书', link: '/zh/whitepaper/', activeMatch: '/zh/whitepaper/' },
            { text: '架构设计', link: '/zh/architecture/overview', activeMatch: '/zh/architecture/' },
            { text: '性能测试', link: '/zh/performance/benchmarks', activeMatch: '/zh/performance/' },
            { text: 'API 参考', link: '/zh/api/spmv', activeMatch: '/zh/api/' },
            { text: '学术引用', link: '/zh/references', activeMatch: '/zh/(references|citation)/' }
          ],
          sidebar: {
            '/zh/': [
              {
                text: '技术白皮书',
                collapsed: false,
                items: [
                  { text: '执行摘要', link: '/zh/whitepaper/' },
                  { text: '设计哲学', link: '/zh/whitepaper/philosophy' },
                  { text: '性能分析', link: '/zh/whitepaper/performance' }
                ]
              },
              {
                text: '快速开始',
                items: [
                  { text: '介绍', link: '/zh/' },
                  { text: '快速开始', link: '/zh/quickstart' },
                  { text: '示例代码', link: '/zh/examples/basic-spmv' }
                ]
              },
              {
                text: '架构设计',
                collapsed: false,
                items: [
                  { text: '系统概览', link: '/zh/architecture/overview' },
                  { text: '执行流水线', link: '/zh/architecture/execution-pipeline' },
                  { text: 'Kernel 选择策略', link: '/zh/architecture/kernel-selection' },
                  { text: '内存布局', link: '/zh/architecture/memory-layout' },
                  { text: '可靠性约束', link: '/zh/architecture/reliability' }
                ]
              },
              {
                text: '性能优化',
                collapsed: false,
                items: [
                  { text: '性能方法学', link: '/zh/performance/methodology' },
                  { text: '基准测试', link: '/zh/performance/benchmarks' },
                  { text: '优化指南', link: '/zh/performance/optimization-guide' }
                ]
              },
              {
                text: 'API 参考',
                collapsed: true,
                items: [
                  { text: 'SpMV 计算', link: '/zh/api/spmv' },
                  { text: 'CSR 矩阵', link: '/zh/api/csr-matrix' },
                  { text: 'ELL 矩阵', link: '/zh/api/ell-matrix' }
                ]
              },
              {
                text: '学术',
                items: [
                  { text: '学术参考', link: '/zh/references' },
                  { text: '引用格式', link: '/zh/citation' },
                  { text: '常见问题', link: '/zh/faq' },
                  { text: '贡献指南', link: '/zh/contributing' }
                ]
              }
            ]
          },
          editLink: {
            pattern: 'https://github.com/AICL-Lab/gpu-spmv/edit/main/docs/:path',
            text: '在 GitHub 上编辑此页'
          },
          docFooter: { prev: '上一页', next: '下一页' },
          outline: { label: '页面导航', level: [2, 3] },
          lastUpdated: { text: '最后更新于' },
          returnToTopLabel: '回到顶部',
          sidebarMenuLabel: '菜单',
          darkModeSwitchLabel: '主题',
          lightModeSwitchTitle: '切换到浅色模式',
          darkModeSwitchTitle: '切换到深色模式'
        }
      },
      en: {
        label: 'English',
        lang: 'en-US',
        link: '/en/',
        description: 'High-Performance CUDA Sparse Matrix-Vector Multiplication Library',
        themeConfig: {
          nav: [
            { text: 'Whitepaper', link: '/en/whitepaper/', activeMatch: '/en/whitepaper/' },
            { text: 'Architecture', link: '/en/architecture/overview', activeMatch: '/en/architecture/' },
            { text: 'Benchmarks', link: '/en/performance/benchmarks', activeMatch: '/en/performance/' },
            { text: 'API Reference', link: '/en/api/spmv', activeMatch: '/en/api/' },
            { text: 'References', link: '/en/references', activeMatch: '/en/(references|citation)/' }
          ],
          sidebar: {
            '/en/': [
              {
                text: 'Technical Whitepaper',
                collapsed: false,
                items: [
                  { text: 'Executive Summary', link: '/en/whitepaper/' },
                  { text: 'Design Philosophy', link: '/en/whitepaper/philosophy' },
                  { text: 'Performance Analysis', link: '/en/whitepaper/performance' }
                ]
              },
              {
                text: 'Getting Started',
                items: [
                  { text: 'Introduction', link: '/en/' },
                  { text: 'Quick Start', link: '/en/quickstart' },
                  { text: 'Examples', link: '/en/examples/basic-spmv' }
                ]
              },
              {
                text: 'Architecture',
                collapsed: false,
                items: [
                  { text: 'System Overview', link: '/en/architecture/overview' },
                  { text: 'Execution Pipeline', link: '/en/architecture/execution-pipeline' },
                  { text: 'Kernel Selection', link: '/en/architecture/kernel-selection' },
                  { text: 'Memory Layout', link: '/en/architecture/memory-layout' },
                  { text: 'Reliability Constraints', link: '/en/architecture/reliability' }
                ]
              },
              {
                text: 'Performance',
                collapsed: false,
                items: [
                  { text: 'Methodology', link: '/en/performance/methodology' },
                  { text: 'Benchmarks', link: '/en/performance/benchmarks' },
                  { text: 'Optimization Guide', link: '/en/performance/optimization-guide' }
                ]
              },
              {
                text: 'API Reference',
                collapsed: true,
                items: [
                  { text: 'SpMV Computation', link: '/en/api/spmv' },
                  { text: 'CSR Matrix', link: '/en/api/csr-matrix' },
                  { text: 'ELL Matrix', link: '/en/api/ell-matrix' }
                ]
              },
              {
                text: 'Academic',
                items: [
                  { text: 'References', link: '/en/references' },
                  { text: 'Citation', link: '/en/citation' },
                  { text: 'FAQ', link: '/en/faq' },
                  { text: 'Contributing', link: '/en/contributing' }
                ]
              }
            ]
          },
          editLink: {
            pattern: 'https://github.com/AICL-Lab/gpu-spmv/edit/main/docs/:path',
            text: 'Edit this page on GitHub'
          },
          outline: { label: 'On This Page', level: [2, 3] }
        }
      }
    },

    themeConfig: {
      logo: {
        light: '/images/brand/logo-mark-light.svg',
        dark: '/images/brand/logo-mark-dark.svg'
      },
      siteTitle: 'GPU SpMV',
      socialLinks: [
        { icon: 'github', link: 'https://github.com/AICL-Lab/gpu-spmv' }
      ],
      search: { provider: 'local' },
      footer: {
        message: 'MIT License',
        copyright: '© 2024-2026 AICL-Lab'
      },
      outline: [2, 3]
    },

  })
)
