# ELL Format Design

## Context

ELL (ELLPACK) format is optimized for sparse matrices with uniform row lengths. The column-major storage enables fully coalesced GPU memory access, making it ideal for certain matrix patterns.

## Goals / Non-Goals

**Goals:**
- Optimize for GPU coalesced memory access
- Support matrices with uniform row lengths efficiently
- Enable high bandwidth utilization

**Non-Goals:**
- Memory efficiency for highly irregular matrices
- Support for extremely variable row lengths (excessive padding waste)

## Decisions

### D1: Column-Major Storage

```
Row-major access pattern (poor):
Thread:   T0      T1      T2
          ↓       ↓       ↓
Address: [row0,k0][row1,k0][row2,k0]  ← Discontiguous!
        [base+0] [base+max_nnz] [base+2*max_nnz]

Column-major access pattern (good):
Thread:   T0      T1      T2
          ↓       ↓       ↓
Address: [row0,k0][row1,k0][row2,k0]  ← Contiguous!
        [base+0] [base+1]   [base+2]
```

**Rationale**: Column-major storage enables adjacent GPU threads to access adjacent memory locations, maximizing memory bandwidth.

### D2: Padding Strategy

```cpp
// -1 indicates padding slot
int col_index = col_indices[k * num_rows + i];
if (col_index >= 0) {
    sum += values[k * num_rows + i] * x[col_index];
}
```

**Rationale**: Using -1 as sentinel value allows efficient padding detection without additional storage.

### D2: Memory Management

Host memory is always owned by the `ELLMatrix` and freed on `ell_destroy()`. Device memory is managed internally via `ell_to_gpu()` / `ell_from_gpu()`.

**Rationale**: Simplifies the public interface by removing ownership flags and device pointers from the public struct.

### D3: Memory Trade-off

| Matrix Pattern | Memory Efficiency |
|----------------|-------------------|
| Uniform rows (all same nnz) | 100% |
| Slight variation | 80-95% |
| High variation | < 50% (use CSR instead) |

**Rationale**: ELL is optimal when row lengths are similar. For highly irregular patterns, CSR with Merge Path kernel is better.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Memory waste with variable row lengths | Use kernel selector to choose CSR for irregular matrices |
| Padding overhead calculation | Compute efficiency metric before format selection |
| Fixed max_nnz_per_row | Reallocate if matrix structure changes |

## Performance Characteristics

- Memory access: Fully coalesced
- Thread divergence: Minimal (uniform work per thread)
- Best for: Matrices with uniform row lengths
- Avoid for: Matrices with high row length variance
