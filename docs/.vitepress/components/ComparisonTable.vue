<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps<{
  lang?: 'zh' | 'en'
}>()

const isZh = computed(() => props.lang !== 'en')

// Comparison data
const comparisons = ref([
  {
    matrix: isZh ? '均匀 100K' : 'Uniform 100K',
    gpuSpmv: '71.5%',
    cuSparse: '68.2%',
    speedup: '1.05×'
  },
  {
    matrix: isZh ? '幂律 100K' : 'Power-Law 100K',
    gpuSpmv: '69.2%',
    cuSparse: '52.1%',
    speedup: '1.33×'
  },
  {
    matrix: isZh ? '真实矩阵 (webbase)' : 'Real-World (webbase)',
    gpuSpmv: '67.8%',
    cuSparse: '61.4%',
    speedup: '1.10×'
  }
])

const advantages = isZh ? [
  '不规则矩阵性能更优（Merge Path 算法）',
  '自动内核选择（无需手动调优）',
  '开源代码（完全透明）'
] : [
  'Better performance on irregular matrices (Merge Path algorithm)',
  'Automatic kernel selection (no manual tuning)',
  'Open source (full transparency)'
]
</script>

<template>
  <div class="comparison-table">
    <table>
      <thead>
        <tr>
          <th>{{ isZh ? '矩阵' : 'Matrix' }}</th>
          <th>GPU SpMV</th>
          <th>cuSPARSE</th>
          <th>{{ isZh ? '加速比' : 'Speedup' }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, index) in comparisons" :key="index">
          <td>{{ row.matrix }}</td>
          <td class="highlight">{{ row.gpuSpmv }}</td>
          <td>{{ row.cuSparse }}</td>
          <td class="speedup">{{ row.speedup }}</td>
        </tr>
      </tbody>
    </table>

    <div class="advantages">
      <h4>{{ isZh ? 'GPU SpMV 优势' : 'GPU SpMV Advantages' }}</h4>
      <ul>
        <li v-for="(adv, index) in advantages" :key="index">{{ adv }}</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.comparison-table {
  padding: 24px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 24px;
}

th, td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--vp-c-border);
}

th {
  background: var(--vp-c-bg);
  font-weight: 600;
  font-size: 14px;
  color: var(--vp-c-text-1);
}

td {
  font-size: 14px;
  color: var(--vp-c-text-2);
}

td.highlight {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

td.speedup {
  color: #10B981;
  font-weight: 600;
}

tr:last-child td {
  border-bottom: none;
}

tr:hover td {
  background: var(--vp-c-bg);
}

.advantages h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: var(--vp-c-text-1);
}

.advantages ul {
  margin: 0;
  padding-left: 20px;
}

.advantages li {
  margin: 8px 0;
  font-size: 14px;
  color: var(--vp-c-text-2);
}
</style>
