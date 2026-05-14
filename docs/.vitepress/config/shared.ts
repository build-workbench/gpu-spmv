import { defineConfig } from 'vitepress'

export const shared = defineConfig({
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
        content:
          'High-Performance CUDA Sparse Matrix-Vector Multiplication Library'
      }
    ],
    [
      'meta',
      { property: 'og:image', content: '/images/og-image.svg' }
    ],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'GPU SpMV' }],
    ['meta', { name: 'twitter:description', content: 'High-Performance CUDA Sparse Matrix-Vector Multiplication Library' }],
    ['meta', { name: 'twitter:image', content: '/images/og-image.svg' }],
    [
      'link',
      { rel: 'icon', href: '/images/favicon.svg', type: 'image/svg+xml' }
    ],
    [
      'link',
      {
        rel: 'preconnect',
        href: 'https://fonts.googleapis.com'
      }
    ],
    [
      'link',
      {
        rel: 'preconnect',
        href: 'https://fonts.gstatic.com',
        crossorigin: ''
      }
    ],
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

  themeConfig: {
    logo: '/images/logo.svg',
    siteTitle: 'GPU SpMV',

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/gpu-spmv' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'MIT License',
      copyright: '© 2024-2026 LessUp'
    }
  }
})
