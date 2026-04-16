# AGENTS.md - AI Agent Workflow Guidelines

> **Purpose**: This file provides instructions for AI coding assistants working on this repository.

---

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the `/specs` directory as the **Single Source of Truth**.

---

## Directory Structure

### Specification Documents

| Directory | Content | Purpose |
|-----------|---------|---------|
| `/specs/product/` | Product Requirements Documents (PRDs) | Feature definitions and acceptance criteria |
| `/specs/rfc/` | Request for Comments (RFCs) | Technical design documents and architecture decisions |
| `/specs/api/` | API Specifications | Interface definitions (human-readable and machine-readable) |
| `/specs/db/` | Database Schema Specs | Data model definitions (if applicable) |
| `/specs/testing/` | BDD Test Specifications | Property-based testing requirements and test cases |

### Documentation

| Directory | Content |
|-----------|---------|
| `/docs/` | User and developer documentation |
| `/docs/setup/` | Environment setup guides |
| `/docs/tutorials/` | User tutorials and how-to guides |
| `/docs/architecture/` | High-level architecture diagrams and descriptions |
| `/docs/assets/` | Static assets (images, diagrams, logos) |

---

## AI Agent Workflow Instructions

**CRITICAL: When you (AI agent) are asked to develop a new feature, modify existing functionality, or fix a bug, you MUST strictly follow this workflow. DO NOT skip any steps.**

### Step 1: Review Specs (MANDATORY)

1. **Read relevant specs first**:
   - Check `/specs/product/` for feature requirements
   - Check `/specs/rfc/` for technical design decisions
   - Check `/specs/api/` for interface definitions
   - Check `/specs/testing/` for test requirements

2. **Identify conflicts**:
   - If the user's request conflicts with existing specs, **STOP immediately**
   - Point out the conflict clearly
   - Ask the user whether specs should be updated first

3. **Document your findings**:
   - List which spec files are relevant
   - Note any gaps or ambiguities

### Step 2: Spec-First Update (MANDATORY)

1. **For new features**:
   - **Propose spec changes FIRST** before writing any code
   - Create or update files in `/specs/` as appropriate:
     - New product requirement → `/specs/product/`
     - New technical design → `/specs/rfc/`
     - New API endpoint → `/specs/api/`
     - New test requirements → `/specs/testing/`

2. **Wait for confirmation**:
   - Present the spec changes to the user
   - **Do not proceed to coding until user confirms the specs**

3. **For bug fixes**:
   - If the bug reveals a gap in specs, update the specs first
   - Add test cases to `/specs/testing/` to prevent regression

### Step 3: Implementation (Follow Specs 100%)

1. **Code according to specs**:
   - Variable naming must match spec definitions
   - API paths, data types, status codes must exactly match `/specs/api/`
   - Architecture patterns must follow `/specs/rfc/`

2. **No gold-plating**:
   - **Do NOT add features not defined in specs**
   - If you think of a useful enhancement, document it as a suggestion but don't implement it
   - Ask the user if they want to add it to specs first

3. **Error handling**:
   - Follow error handling conventions defined in specs
   - Use existing error codes and patterns

### Step 4: Test Against Specs (MANDATORY)

1. **Write tests based on specs**:
   - Use `/specs/testing/` as the source for test cases
   - Ensure all acceptance criteria from `/specs/product/` are covered
   - Property-based tests must run minimum 100 iterations

2. **Test coverage**:
   - Cover all boundary conditions described in specs
   - Test edge cases explicitly mentioned in requirements

3. **Run tests**:
   ```bash
   # Configure and build
   cmake --preset default && cmake --build --preset default
   
   # Run tests
   ctest --preset default
   ```

---

## Code Generation Rules

### API Changes
- **Any externally exposed API change MUST update `/specs/api/public-api.md`**
- This includes:
  - New functions or classes
  - Modified function signatures
  - Changed data structures
  - New error codes

### Architecture Decisions
- **When uncertain about technical details, consult `/specs/rfc/`**
- Do not invent design patterns independently
- Follow existing architectural conventions documented in specs

### Code Style
- Follow project conventions defined in `CLAUDE.md`:
  - clang-format enforced (Google style, 4-space indent, 100 col limit)
  - Include order: `"spmv/"` first, then `<cuda*`, then `<`, then everything else
  - Namespace: `spmv::`
  - Commit messages: Conventional Commits

### Testing Conventions
- Google Test framework
- Property-based tests use 100 iterations with random matrices
- Test files follow naming: `*_test.cpp` or `*_tests.cu`

---

## Common Pitfalls

### GPU/CI Environment
- **CI has no GPU** — tests requiring CUDA devices will fail in CI
- `benchmarks/main.cu` and `pagerank.cu` exit early with error if no CUDA device found
- Use CPU-only tests for CI validation

### Build System
- **No Makefile** — all builds via CMake presets
- Three presets: `default` (Debug), `release` (Release), `minimal` (sm_80 only)
- CPU-only configure: `cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF`

### Resource Management
- Use RAII patterns (`CudaBuffer<T>`, `SpMVExecutionContext`)
- Never use raw `cudaMalloc`/`cudaFree` in new code
- Follow existing error handling macros (`CUDA_CHECK`, `CUDA_CHECK_MALLOC`, etc.)

---

## Workflow Example

### User Request: "Add support for COO format"

**AI Agent Should**:

1. ✅ **Review existing specs**:
   - Read `/specs/product/spmv-gpu.md` to understand current formats (CSR, ELL)
   - Read `/specs/rfc/0001-core-architecture.md` to understand architecture
   - Check `/specs/api/public-api.md` for API patterns

2. ✅ **Propose spec updates**:
   - Create `/specs/product/coo-format.md` with requirements
   - Create `/specs/rfc/0003-coo-format-support.md` with technical design
   - Update `/specs/api/public-api.md` with new COO APIs
   - Update `/specs/testing/property-tests.feature` with COO test properties

3. ⏸️ **Wait for user confirmation on specs**

4. ✅ **Implement after approval**:
   - Create `include/spmv/coo_matrix.h` following API spec
   - Implement COO data structure and operations
   - Implement COO SpMV kernel
   - Add COO-to-CSR/ELL conversion

5. ✅ **Write tests**:
   - Add COO property tests per `/specs/testing/`
   - Run 100 iterations with random matrices
   - Ensure all acceptance criteria met

---

## Why These Rules Exist

| Rule | Reason |
|------|--------|
| **Read specs first** | Prevents AI hallucination and anchors thinking to defined requirements |
| **Spec-first updates** | Ensures documentation and code stay synchronized |
| **No gold-plating** | Keeps implementation focused on user needs, reduces scope creep |
| **Test against specs** | Validates implementation meets acceptance criteria |
| **Update API specs** | Maintains single source of truth for external interfaces |

---

## Quick Reference

```
Workflow: Review Specs → Update Specs → Get Approval → Implement → Test
Specs: /specs/{product, rfc, api, db, testing}/
Docs: /docs/{setup, tutorials, architecture, assets}/
Build: cmake --preset default && cmake --build --preset default
Test: ctest --preset default
Style: See CLAUDE.md
```
