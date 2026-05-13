import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { shared } from './config/shared'
import { zh } from './config/zh'
import { en } from './config/en'

export default withMermaid(
  defineConfig({
    ...shared,
    locales: {
      zh: { label: '简体中文', lang: 'zh-CN', ...zh },
      en: { label: 'English', lang: 'en-US', ...en }
    },
    mermaid: {
      // Mermaid configuration
    }
  })
)
