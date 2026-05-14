<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useData } from 'vitepress'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { Bar } from 'vue-chartjs'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const props = defineProps<{
  title?: string
}>()

const { isDark } = useData()

const colors = computed(() => isDark.value
  ? { text: '#c9d1d9', subtext: '#8b949e', grid: 'rgba(48, 54, 61, 0.5)' }
  : { text: '#24292f', subtext: '#57606a', grid: 'rgba(208, 215, 222, 0.5)' }
)

const chartData = ref({
  labels: ['Diagonal', 'Uniform', 'Power-Law', 'Band'],
  datasets: [
    {
      label: 'Scalar CSR',
      data: [37.2, 41.5, 32.1, 28.4],
      backgroundColor: 'rgba(139, 92, 246, 0.8)',
      borderColor: 'rgba(139, 92, 246, 1)',
      borderWidth: 1
    },
    {
      label: 'Vector CSR',
      data: [69.1, 71.8, 45.6, 64.9],
      backgroundColor: 'rgba(59, 130, 246, 0.8)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 1
    },
    {
      label: 'Merge Path',
      data: [72.4, 70.9, 69.2, 58.1],
      backgroundColor: 'rgba(16, 185, 129, 0.8)',
      borderColor: 'rgba(16, 185, 129, 1)',
      borderWidth: 1
    },
    {
      label: 'ELL Kernel',
      data: [74.8, 82.3, 34.7, 41.2],
      backgroundColor: 'rgba(245, 158, 11, 0.8)',
      borderColor: 'rgba(245, 158, 11, 1)',
      borderWidth: 1
    }
  ]
})

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom' as const,
      labels: {
        color: colors.value.text,
        padding: 20,
        font: { size: 12 }
      }
    },
    title: {
      display: !!props.title,
      text: props.title || '',
      color: colors.value.text,
      font: { size: 16, weight: 'bold' as const }
    },
    tooltip: {
      callbacks: {
        label: (context: any) => `${context.dataset.label}: ${context.raw}%`
      }
    }
  },
  scales: {
    x: {
      ticks: { color: colors.value.subtext },
      grid: { color: colors.value.grid }
    },
    y: {
      beginAtZero: true,
      max: 100,
      ticks: {
        color: colors.value.subtext,
        callback: (value: any) => value + '%'
      },
      grid: { color: colors.value.grid },
      title: {
        display: true,
        text: 'Bandwidth Utilization (%)',
        color: colors.value.subtext
      }
    }
  }
}))
</script>

<template>
  <div class="performance-chart">
    <Bar :data="chartData" :options="chartOptions" />
  </div>
</template>

<style scoped>
.performance-chart {
  width: 100%;
  height: 400px;
  padding: 16px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
}
</style>
