<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    variant?: 'overview-zh' | 'overview-en'
  }>(),
  {
    variant: 'overview-en'
  }
)

const labels = computed(() =>
  props.variant === 'overview-zh'
    ? {
        input: '稀疏矩阵',
        analysis: '矩阵分析',
        decision: '内核选择',
        execution: 'GPU 执行',
        result: '结果验证'
      }
    : {
        input: 'Sparse Matrix',
        analysis: 'Matrix Analysis',
        decision: 'Kernel Choice',
        execution: 'GPU Execution',
        result: 'Result Validation'
      }
)
</script>

<template>
  <div class="spmv-architecture-shell spmv-surface-card">
    <svg viewBox="0 0 820 260" class="spmv-architecture-canvas" role="img" aria-label="SpMV architecture">
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
        </marker>
      </defs>

      <path class="spmv-link" d="M130 90 H240" marker-end="url(#arrow)" />
      <path class="spmv-link" d="M370 90 H480" marker-end="url(#arrow)" />
      <path class="spmv-link" d="M610 90 H720" marker-end="url(#arrow)" />
      <path class="spmv-link" d="M540 150 Q540 220 300 220 Q120 220 120 150" marker-end="url(#arrow)" />

      <g transform="translate(20 50)">
        <rect class="spmv-node" width="110" height="80" rx="20" />
        <text class="spmv-node-text" x="55" y="34" text-anchor="middle">{{ labels.input }}</text>
        <text class="spmv-node-caption" x="55" y="56" text-anchor="middle">CSR / ELL</text>
      </g>
      <g transform="translate(250 50)">
        <rect class="spmv-node" width="120" height="80" rx="20" />
        <text class="spmv-node-text" x="60" y="34" text-anchor="middle">{{ labels.analysis }}</text>
        <text class="spmv-node-caption" x="60" y="56" text-anchor="middle">avg_nnz / skewness</text>
      </g>
      <g transform="translate(490 50)">
        <rect class="spmv-node" width="120" height="80" rx="20" />
        <text class="spmv-node-text" x="60" y="34" text-anchor="middle">{{ labels.decision }}</text>
        <text class="spmv-node-caption" x="60" y="56" text-anchor="middle">Scalar / Vector / Merge</text>
      </g>
      <g transform="translate(690 50)">
        <rect class="spmv-node" width="110" height="80" rx="20" />
        <text class="spmv-node-text" x="55" y="34" text-anchor="middle">{{ labels.execution }}</text>
        <text class="spmv-node-caption" x="55" y="56" text-anchor="middle">CUDA kernel</text>
      </g>
      <g transform="translate(350 170)">
        <rect class="spmv-node" width="140" height="70" rx="20" />
        <text class="spmv-node-text" x="70" y="30" text-anchor="middle">{{ labels.result }}</text>
        <text class="spmv-node-caption" x="70" y="51" text-anchor="middle">Accuracy + bandwidth</text>
      </g>
    </svg>
  </div>
</template>
