<script setup lang="ts">
import { ref } from 'vue'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { Line } from 'vue-chartjs'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const props = defineProps<{
  title?: string
}>()

// Performance trend by matrix size
const chartData = ref({
  labels: ['10K × 10K', '100K × 100K', '1M × 1M', '10M × 10M'],
  datasets: [
    {
      label: 'Scalar CSR',
      data: [42.1, 36.7, 34.8, 33.2],
      borderColor: 'rgba(139, 92, 246, 1)',
      backgroundColor: 'rgba(139, 92, 246, 0.1)',
      tension: 0.3,
      fill: false
    },
    {
      label: 'Vector CSR',
      data: [70.2, 68.7, 65.5, 62.1],
      borderColor: 'rgba(59, 130, 246, 1)',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      tension: 0.3,
      fill: false
    },
    {
      label: 'Merge Path',
      data: [68.5, 71.5, 70.8, 69.4],
      borderColor: 'rgba(16, 185, 129, 1)',
      backgroundColor: 'rgba(16, 185, 129, 0.1)',
      tension: 0.3,
      fill: false
    },
    {
      label: 'ELL Kernel',
      data: [78.3, 73.7, 71.2, 68.9],
      borderColor: 'rgba(245, 158, 11, 1)',
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      tension: 0.3,
      fill: false
    }
  ]
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom' as const,
      labels: {
        color: '#c9d1d9',
        padding: 20,
        font: {
          size: 12
        }
      }
    },
    title: {
      display: !!props.title,
      text: props.title || '',
      color: '#c9d1d9',
      font: {
        size: 16,
        weight: 'bold' as const
      }
    },
    tooltip: {
      callbacks: {
        label: function(context: any) {
          return `${context.dataset.label}: ${context.raw}%`
        }
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: '#8b949e'
      },
      grid: {
        color: 'rgba(48, 54, 61, 0.5)'
      }
    },
    y: {
      beginAtZero: true,
      max: 100,
      ticks: {
        color: '#8b949e',
        callback: function(value: any) {
          return value + '%'
        }
      },
      grid: {
        color: 'rgba(48, 54, 61, 0.5)'
      },
      title: {
        display: true,
        text: 'Bandwidth Utilization (%)',
        color: '#8b949e'
      }
    }
  }
}
</script>

<template>
  <div class="trend-chart">
    <Line :data="chartData" :options="chartOptions" />
  </div>
</template>

<style scoped>
.trend-chart {
  width: 100%;
  height: 400px;
  padding: 16px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
}
</style>
