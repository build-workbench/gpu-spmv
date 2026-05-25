# Reliability and Engineering Constraints

## Where reliability comes from

Reliability in this project is not “it seems to run.” It comes from three lines of evidence working together:

1. **Explicit resource lifetime** through `CudaBuffer<T>` and execution-context abstractions instead of raw `cudaMalloc` / `cudaFree`.
2. **Explicit error semantics** through `SpMVError` and CUDA checking macros.
3. **Tests that stay close to the code** through CPU reference paths and focused regression coverage.

## Why this matters for a showcase project

Interviewers and open-source readers trust a project more when it can answer:

- How are resources released?
- How are failures surfaced?
- How should benchmark evidence be interpreted?
- How are design changes traced?

## Reliability Checklist

- RAII wrappers for GPU resources
- CPU reference paths for cross-checking
- property tests with at least 100 iterations
- GitHub Pages used to publish architecture and references, not only marketing copy
