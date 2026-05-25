import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = process.cwd()
const canonicalRepo = 'AICL-Lab/gpu-spmv'

function collectTextFiles(dirPath) {
  const entries = readdirSync(dirPath, { withFileTypes: true })
  const files = []

  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name)
    if (entry.isDirectory()) {
      files.push(...collectTextFiles(fullPath))
      continue
    }
    if (/\.(md|ts|yml|svg)$/.test(entry.name)) {
      files.push(fullPath)
    }
  }

  return files
}

const files = {
  readme: join(root, '..', 'README.md'),
  readmeZh: join(root, '..', 'README.zh-CN.md'),
  config: join(root, '.vitepress', 'config.ts'),
  pages: join(root, '..', '.github', 'workflows', 'pages.yml'),
  index: join(root, 'index.md'),
  themeIndex: join(root, '.vitepress', 'theme', 'index.ts'),
  zhHome: join(root, 'zh', 'index.md'),
  enHome: join(root, 'en', 'index.md'),
  zhWhitepaper: join(root, 'zh', 'whitepaper', 'index.md'),
  enWhitepaper: join(root, 'en', 'whitepaper', 'index.md')
}

const contents = Object.fromEntries(
  Object.entries(files).map(([key, filePath]) => [key, readFileSync(filePath, 'utf8')])
)

const failures = []

if (!contents.config.includes(canonicalRepo)) {
  failures.push('config missing canonical repo')
}

if (!contents.pages.includes("github.repository == 'AICL-Lab/gpu-spmv'")) {
  failures.push('pages workflow missing canonical repo guard')
}

if (/LessUp\/gpu-spmv|github\.com\/LessUp/.test(Object.values(contents).join('\n'))) {
  failures.push('legacy LessUp repo references still present')
}

if (/useRouter\(|router\.go\('\/(zh|en)\//.test(contents.index)) {
  failures.push('root docs index still auto-redirects by locale')
}

const requiredThemeFiles = [
  join(root, '.vitepress', 'theme', 'Layout.vue'),
  join(root, '.vitepress', 'theme', 'styles', 'tokens.css'),
  join(root, '.vitepress', 'theme', 'styles', 'base.css'),
  join(root, '.vitepress', 'theme', 'styles', 'home.css'),
  join(root, '.vitepress', 'theme', 'styles', 'paper.css'),
  join(root, '.vitepress', 'theme', 'styles', 'citation.css'),
  join(root, '.vitepress', 'theme', 'styles', 'diagram.css'),
  join(root, '.vitepress', 'theme', 'components', 'HeroEvidence.vue'),
  join(root, '.vitepress', 'theme', 'components', 'MetricStrip.vue'),
  join(root, '.vitepress', 'theme', 'components', 'WhitepaperSection.vue'),
  join(root, '.vitepress', 'theme', 'components', 'ArchitectureCanvas.vue'),
  join(root, '.vitepress', 'theme', 'components', 'CitationGrid.vue'),
  join(root, '.vitepress', 'theme', 'components', 'ThemeAwareArt.vue'),
  join(root, '.vitepress', 'theme', 'components', 'CalloutPanel.vue'),
  join(root, '.vitepress', 'data', 'site.ts')
]

const requiredAssetFiles = [
  join(root, 'public', 'images', 'brand', 'logo-mark-light.svg'),
  join(root, 'public', 'images', 'brand', 'logo-mark-dark.svg'),
  join(root, 'public', 'images', 'social', 'og-light.svg'),
  join(root, 'public', 'images', 'social', 'og-dark.svg')
]

const requiredContentFiles = [
  join(root, '.vitepress', 'data', 'references.ts'),
  join(root, '.vitepress', 'data', 'benchmarks.ts'),
  join(root, 'zh', 'architecture', 'execution-pipeline.md'),
  join(root, 'en', 'architecture', 'execution-pipeline.md'),
  join(root, 'zh', 'architecture', 'reliability.md'),
  join(root, 'en', 'architecture', 'reliability.md'),
  join(root, 'zh', 'performance', 'methodology.md'),
  join(root, 'en', 'performance', 'methodology.md')
]

for (const filePath of [...requiredThemeFiles, ...requiredAssetFiles, ...requiredContentFiles]) {
  if (!existsSync(filePath)) {
    failures.push(`missing theme file: ${filePath.replace(`${root}/`, '')}`)
  }
}

const themeIndexChecks = [
  'HeroEvidence',
  'MetricStrip',
  'WhitepaperSection',
  'ArchitectureCanvas',
  'CitationGrid',
  'ThemeAwareArt',
  'CalloutPanel'
]

for (const token of themeIndexChecks) {
  if (!contents.themeIndex.includes(token)) {
    failures.push(`theme index missing component registration: ${token}`)
  }
}

if (!contents.zhHome.includes('<HeroEvidence') || !contents.zhHome.includes('<ArchitectureCanvas')) {
  failures.push('zh homepage has not been rebuilt with theme components')
}

if (!contents.enHome.includes('<HeroEvidence') || !contents.enHome.includes('<ArchitectureCanvas')) {
  failures.push('en homepage has not been rebuilt with theme components')
}

if (!contents.zhWhitepaper.includes('<CalloutPanel')) {
  failures.push('zh whitepaper landing page missing positioning callout')
}

if (!contents.enWhitepaper.includes('<CalloutPanel')) {
  failures.push('en whitepaper landing page missing positioning callout')
}

if (!contents.config.includes("link: '/zh/references'")) {
  failures.push('zh nav missing references entry')
}

if (!contents.config.includes("link: '/en/references'")) {
  failures.push('en nav missing references entry')
}

if (!contents.config.includes("light: '/images/brand/logo-mark-light.svg'")) {
  failures.push('config missing light-mode logo asset')
}

if (!contents.config.includes("dark: '/images/brand/logo-mark-dark.svg'")) {
  failures.push('config missing dark-mode logo asset')
}

if (!contents.config.includes("`${base}images/social/og-dark.svg`")) {
  failures.push('config missing social og image upgrade')
}

if (!contents.config.includes("link: '/zh/architecture/execution-pipeline'")) {
  failures.push('zh sidebar missing execution pipeline entry')
}

if (!contents.config.includes("link: '/en/architecture/execution-pipeline'")) {
  failures.push('en sidebar missing execution pipeline entry')
}

if (!contents.config.includes("link: '/zh/performance/methodology'")) {
  failures.push('zh sidebar missing methodology entry')
}

if (!contents.config.includes("link: '/en/performance/methodology'")) {
  failures.push('en sidebar missing methodology entry')
}

const docsCorpus = collectTextFiles(join(root, 'zh'))
  .concat(collectTextFiles(join(root, 'en')))
  .concat([join(root, '..', 'README.md'), join(root, '..', 'README.zh-CN.md')])
  .map((filePath) => readFileSync(filePath, 'utf8'))
  .join('\n')

if (/LessUp\/gpu-spmv|github\.com\/LessUp|lessup\.github\.io\/gpu-spmv/.test(docsCorpus)) {
  failures.push('legacy LessUp references still exist in docs corpus')
}

if (failures.length > 0) {
  console.error('verify-site failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('verify-site: ok')
