import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = process.cwd()
const canonicalRepo = 'AICL-Lab/gpu-spmv'

const files = {
  readme: join(root, '..', 'README.md'),
  config: join(root, '.vitepress', 'config.ts'),
  pages: join(root, '..', '.github', 'workflows', 'pages.yml'),
  index: join(root, 'index.md')
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

if (failures.length > 0) {
  console.error('verify-site failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('verify-site: ok')
