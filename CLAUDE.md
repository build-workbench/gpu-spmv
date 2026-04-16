# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the `/specs` directory as the Single Source of Truth.

### Directory Context
- `/specs/product/`: Product feature definitions and acceptance criteria (PRDs)
- `/specs/rfc/`: Technical design documents (RFCs)
- `/specs/api/`: API interface specifications (OpenAPI-style documentation)
- `/specs/testing/`: BDD test specifications and property-based testing requirements

### AI Agent Workflow Instructions

**When you (AI) are asked to develop a new feature, modify existing functionality, or fix a bug, you MUST strictly follow this workflow without skipping any steps:**

#### Step 1: Review Specs
- First, read the relevant documentation in `/specs` (product requirements, RFCs, API definitions)
- If the user's request conflicts with existing specs, STOP coding immediately and point out the conflict, asking whether specs should be updated first

#### Step 2: Spec-First Update
- For new features or changes affecting interfaces/data structures, **propose modifications to spec documents FIRST** (e.g., updating RFCs or API docs)
- Wait for user confirmation on spec changes before entering the coding phase

#### Step 3: Implementation
- When writing code, **100% comply with spec definitions** (including variable naming, API paths, data types, status codes, etc.)
- **Do not add features not defined in specs** (No Gold-Plating)

#### Step 4: Test Against Specs
- Write unit and integration tests based on acceptance criteria in `/specs`
- Ensure test cases cover all boundary conditions described in specs

### Code Generation Rules
- Any externally exposed API changes must simultaneously modify `/specs/api/public-api.md`
- When uncertain about technical details, consult `/specs/rfc/` for architectural conventions—do not invent design patterns independently

---

## Project

GPU-accelerated Sparse Matrix-Vector Multiplication (SpMV) library in C++/CUDA. Supports CSR and ELL formats with multiple kernel strategies and auto-selection. Includes PageRank and benchmarking.

## Build & Test

```bash
# Configure + build
cmake --preset default && cmake --build --preset default   # Debug
cmake --preset release && cmake --build --preset release   # Release

# CPU-only configure (no CUDA device needed)
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF

# Test
ctest --preset default
./build/spmv_tests              # direct
```

No Makefile — all via CMake presets. Three presets: `default` (Debug), `release` (Release), `minimal` (sm_80 only).

## Code Style

- clang-format enforced in CI (`.clang-format` based on Google style, 4-space indent, 100 col limit)
- Include order: `"spmv/"` first, then `<cuda*`, then `<`, then everything else
- Namespace: `spmv::`
- Error handling: `SpMVError` enum, `CUDA_CHECK`/`CUDA_CHECK_MALLOC`/`CUDA_CHECK_MEMCPY` macros, RAII via `CudaBuffer<T>`

## Conventions

- Commit messages: Conventional Commits (`feat:`, `fix:`, `perf:`, `build:`, `refactor:`, etc.)
- Tests: Google Test, property-based tests use 100 iterations with random matrices
- Design specs live in `/specs/` directory (requirements, rfc, api, testing)

## Key Gotchas

- CI has no GPU — tests requiring CUDA devices will fail in CI
- `benchmarks/main.cu` and `pagerank.cu` exit early with error if no CUDA device found
- The `minimal` preset hardcodes sm_80; other presets let CMake auto-detect architectures
