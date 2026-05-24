import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import HeroEvidence from './components/HeroEvidence.vue'
import MetricStrip from './components/MetricStrip.vue'
import WhitepaperSection from './components/WhitepaperSection.vue'
import ArchitectureCanvas from './components/ArchitectureCanvas.vue'
import CitationGrid from './components/CitationGrid.vue'
import ThemeAwareArt from './components/ThemeAwareArt.vue'
import CalloutPanel from './components/CalloutPanel.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('HeroEvidence', HeroEvidence)
    app.component('MetricStrip', MetricStrip)
    app.component('WhitepaperSection', WhitepaperSection)
    app.component('ArchitectureCanvas', ArchitectureCanvas)
    app.component('CitationGrid', CitationGrid)
    app.component('ThemeAwareArt', ThemeAwareArt)
    app.component('CalloutPanel', CalloutPanel)
  }
}
