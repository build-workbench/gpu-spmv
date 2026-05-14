<script setup lang="ts">
import { ref, computed } from 'vue'

// Kernel selection inputs
const avgNnz = ref<number>(10)
const skewness = ref<number>(5)

// Computed kernel selection
const selectedKernel = computed(() => {
  if (avgNnz.value < 4) {
    return { name: 'Scalar CSR', color: '#8B5CF6', reason: 'avg_nnz < 4' }
  } else if (skewness.value < 10) {
    return { name: 'Vector CSR', color: '#3B82F6', reason: 'skewness < 10' }
  } else {
    return { name: 'Merge Path', color: '#10B981', reason: 'skewness >= 10' }
  }
})

// Performance estimate
const performanceEstimate = computed(() => {
  if (avgNnz.value < 4) return '35-45%'
  if (skewness.value < 10) return '68-72%'
  return '69-72%'
})
</script>

<template>
  <div class="kernel-selector">
    <div class="selector-header">
      <h3>Interactive Kernel Selection</h3>
      <p>Adjust matrix characteristics to see which kernel will be selected</p>
    </div>

    <div class="selector-controls">
      <div class="control-group">
        <label>Average Non-zeros per Row (avg_nnz)</label>
        <div class="slider-container">
          <input
            type="range"
            v-model.number="avgNnz"
            min="1"
            max="50"
            step="1"
          />
          <span class="slider-value">{{ avgNnz }}</span>
        </div>
      </div>

      <div class="control-group">
        <label>Row Length Skewness</label>
        <div class="slider-container">
          <input
            type="range"
            v-model.number="skewness"
            min="0"
            max="30"
            step="1"
          />
          <span class="slider-value">{{ skewness }}</span>
        </div>
      </div>
    </div>

    <div class="selector-result">
      <div class="result-card" :style="{ borderColor: selectedKernel.color }">
        <div class="result-label">Selected Kernel</div>
        <div class="result-value" :style="{ color: selectedKernel.color }">
          {{ selectedKernel.name }}
        </div>
        <div class="result-reason">{{ selectedKernel.reason }}</div>
      </div>

      <div class="result-card">
        <div class="result-label">Expected Performance</div>
        <div class="result-value">{{ performanceEstimate }}</div>
        <div class="result-reason">bandwidth utilization</div>
      </div>
    </div>

    <div class="selector-diagram">
      <pre class="diagram-code">
if (avg_nnz < {{ avgNnz < 4 ? avgNnz : 4 }}) → Scalar CSR
else if (skewness < {{ skewness < 10 ? skewness : 10 }}) → Vector CSR
else → Merge Path
      </pre>
    </div>
  </div>
</template>

<style scoped>
.kernel-selector {
  padding: 24px;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  border: 1px solid var(--vp-c-border);
}

.selector-header {
  margin-bottom: 24px;
}

.selector-header h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  color: var(--vp-c-text-1);
}

.selector-header p {
  margin: 0;
  font-size: 14px;
  color: var(--vp-c-text-2);
}

.selector-controls {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
  margin-bottom: 24px;
}

@media (max-width: 640px) {
  .selector-controls {
    grid-template-columns: 1fr;
  }
}

.control-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider-container input[type="range"] {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-border);
  appearance: none;
  cursor: pointer;
}

.slider-container input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
}

.slider-value {
  min-width: 40px;
  padding: 4px 8px;
  border-radius: 4px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-border);
  font-size: 14px;
  font-family: monospace;
  color: var(--vp-c-text-1);
  text-align: center;
}

.selector-result {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.result-card {
  padding: 16px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 2px solid var(--vp-c-border);
  text-align: center;
}

.result-label {
  font-size: 12px;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.result-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  margin-bottom: 4px;
}

.result-reason {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-family: monospace;
}

.selector-diagram {
  padding: 16px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-border);
}

.diagram-code {
  margin: 0;
  padding: 0;
  font-size: 13px;
  font-family: monospace;
  color: var(--vp-c-text-2);
  white-space: pre;
}
</style>
