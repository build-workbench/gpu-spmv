import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import './style.css'

// Import custom components
import PerformanceChart from '../components/PerformanceChart.vue'
import TrendChart from '../components/TrendChart.vue'
import KernelSelector from '../components/KernelSelector.vue'
import ComparisonTable from '../components/ComparisonTable.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {})
  },
  enhanceApp({ app }) {
    // Register custom components globally
    app.component('PerformanceChart', PerformanceChart)
    app.component('TrendChart', TrendChart)
    app.component('KernelSelector', KernelSelector)
    app.component('ComparisonTable', ComparisonTable)
  }
} satisfies Theme
