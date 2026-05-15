# Spec-Driven Development

GPU SpMV uses **OpenSpec** specification-driven development. All features are defined in specs before implementation.

## What is OpenSpec?

OpenSpec is a structured specification system where specs are the single source of truth:

```
openspec/
├── specs/           # Feature specs (single source of truth)
│   ├── csr-format/
│   │   ├── spec.md      # Interface contract
│   │   └── design.md    # Design decisions
│   ├── ell-format/
│   ├── spmv-kernels/
│   ├── public-api/
│   └── ...
└── changes/         # Change proposals
    ├── active/      # In-progress changes
    └── archive/     # Completed changes
```

## Spec Example

### CSR Format Spec (excerpt)

```yaml
# openspec/specs/csr-format/spec.md

Feature: CSR Sparse Matrix Format
Status: STABLE

Interface:
  - csr_create(num_rows, num_cols, nnz) -> CSRMatrix*
  - csr_destroy(mat)
  - csr_to_gpu(mat) -> int
  - csr_from_gpu(mat) -> int

Invariants:
  - mat->nnz == mat->row_ptrs[mat->num_rows]
  - mat->row_ptrs[i] <= mat->row_ptrs[i+1]
  - all indices in col_indices are valid

Test Requirements:
  - Must verify memory leaks
  - Must verify boundary conditions
  - Property tests: >= 100 iterations
```

## Change History

| Change | Date | Impact | Status |
|:-------|:-----|:-------|:------:|
| CSR Format Implementation | 2025-01-15 | Core data structure | ✅ |
| ELL Format Support | 2025-02-10 | Multi-format | ✅ |
| SpMV Kernel Optimization | 2025-02-20 | Performance | ✅ |
| Kernel Auto-Selection | 2025-03-01 | Usability | ✅ |
| Benchmark Framework | 2025-03-05 | Verifiability | ✅ |
| PageRank Application | 2025-03-10 | Application layer | ✅ |
| Project Completion | 2026-04-01 | Overall quality | ✅ |

## Why Spec-Driven?

### 1. Traceability

Every design decision is documented.

### 2. Verifiability

Specs serve as test contracts.

### 3. Maintainability

New contributors quickly understand the design.

### 4. Consistency

Spec-driven development prevents implementation drift.

## Interview Value

Demonstrating Spec-Driven Development in interviews:

1. **Professional methodology**: Shows software engineering best practices
2. **Documentation skills**: Spec docs show technical writing ability
3. **Quality mindset**: Test-driven, verifiable
4. **Maintenance thinking**: Considers long-term maintenance

## References

- [OpenSpec Specs](https://github.com/AICL-Lab/gpu-spmv/tree/main/openspec)
- [Architecture Overview](/en/architecture/overview)