# Specifications Index

This directory contains all specification documents for the GPU SpMV project. The project follows **Spec-Driven Development (SDD)**, making these documents the **Single Source of Truth** for all implementation decisions.

---

## Directory Structure

```
specs/
├── product/          # Product Requirements Documents (PRDs)
├── rfc/              # Request for Comments (Technical Design)
├── api/              # API Specifications
├── db/               # Database Schema Specs (if applicable)
└── testing/          # BDD Test Specifications
```

---

## 📁 Product Requirements (`/specs/product/`)

Product requirements define **what** features the system should have and their acceptance criteria.

| Document | Description | Status |
|----------|-------------|--------|
| [spmv-gpu.md](product/spmv-gpu.md) | Complete SpMV feature specifications including CSR/ELL formats, kernels, PageRank, and benchmarking | ✅ v1.0.0 |

### Content Overview

The SpMV GPU product requirements include:
- **REQ-1**: CSR Format Storage
- **REQ-2**: ELL Format Storage
- **REQ-3**: Basic SpMV CUDA Kernel
- **REQ-4**: Load Balancing Optimization
- **REQ-5**: Bandwidth Optimization
- **REQ-6**: Performance Benchmarking
- **REQ-7**: PageRank Algorithm
- **REQ-8**: Error Handling & Resource Management

---

## 📁 Technical RFCs (`/specs/rfc/`)

RFCs (Request for Comments) define **how** the system is architected and designed.

| Document | Description | Status |
|----------|-------------|--------|
| [0001-core-architecture.md](rfc/0001-core-architecture.md) | Core system architecture, data structures, kernel designs, and optimization strategies | ✅ v1.0.0 |
| [0002-implementation-tasks.md](rfc/0002-implementation-tasks.md) | Complete implementation task checklist with requirement traceability | ✅ v1.0.0 |

### Content Overview

**RFC 0001 - Core Architecture**:
- Layered architecture (Storage, Compute, Application)
- CSR and ELL matrix format implementations
- Four CUDA kernel designs (Scalar CSR, Vector CSR, Merge Path, ELL)
- Kernel selection strategy
- Bandwidth optimization techniques
- PageRank algorithm implementation
- Error handling with RAII patterns

**RFC 0002 - Implementation Tasks**:
- 12-phase implementation plan
- Task-to-requirement traceability matrix
- Property test mapping
- Development notes and checkpoints

---

## 📁 API Specifications (`/specs/api/`)

API specifications define the public interfaces exposed by the library.

| Document | Description | Status |
|----------|-------------|--------|
| [public-api.md](api/public-api.md) | Complete public API reference including all data structures, functions, and error handling | ✅ v1.0.0 |

### Content Overview

- **Error Handling**: `SpMVError` enum, error strings, CUDA helper macros
- **Data Structures**: `CSRMatrix`, `ELLMatrix`, `SpMVConfig`, `SpMVResult`, `PageRankConfig`, etc.
- **Core Functions**:
  - CSR/ELL matrix operations (create, destroy, convert, serialize)
  - SpMV computation (`spmv_csr`, `spmv_ell`, `spmv_auto_config`)
  - PageRank algorithm (`pagerank`, `pagerank_top_k`)
  - Benchmarking framework
- **Execution Context**: Resource reuse patterns
- **RAII Templates**: `CudaBuffer<T>` guarantees
- **Naming Conventions**: PascalCase, snake_case, UPPER_SNAKE_CASE rules
- **Versioning**: Semantic versioning policy

---

## 📁 Database Schema Specs (`/specs/db/`)

Currently empty. This directory would contain database schema specifications if the project uses persistent storage.

---

## 📁 Testing Specifications (`/specs/testing/`)

Test specifications define **how** the system should be validated.

| Document | Description | Status |
|----------|-------------|--------|
| [property-tests.feature](testing/property-tests.feature) | Complete BDD property test specifications with 16 properties | ✅ v1.0.0 |

### Content Overview

**16 Property Tests**:

| ID | Property | Validates |
|----|----------|-----------|
| P1 | CSR Dense-to-Sparse Round Trip | REQ-1.2 |
| P2 | CSR Element Lookup Correctness | REQ-1.3 |
| P3 | CSR Serialization Round Trip | REQ-1.5 |
| P4 | ELL Dense-to-Sparse Round Trip | REQ-2.2 |
| P5 | ELL Padding Correctness | REQ-2.3 |
| P6 | ELL Column-Major Layout | REQ-2.4 |
| P7 | ELL Serialization Round Trip | REQ-2.5 |
| P8 | SpMV CSR Correctness | REQ-3.1, 3.3 |
| P9 | SpMV ELL Correctness | REQ-3.2, 3.3 |
| P10 | SpMV Dimension Validation | REQ-3.5, 8.5 |
| P11 | Kernel Selector Validity | REQ-4.5 |
| P12 | Bandwidth Metrics Validity | REQ-5.5 |
| P13 | Benchmark Metrics Completeness | REQ-6.1, 6.3 |
| P14 | Benchmark JSON Round Trip | REQ-6.5 |
| P15 | PageRank Score Invariants | REQ-7.1, 7.2 |
| P16 | PageRank Top-K Ordering | REQ-7.5 |

Each property test includes:
- Gherkin-style BDD specification
- C++ test implementation template
- Requirement traceability
- Minimum 100 iterations with random matrices

---

## How to Use Specs

### For Developers

1. **Before implementing a feature**: Read relevant specs
2. **When adding features**: Update specs first, get approval, then implement
3. **When fixing bugs**: Check if specs need updates to prevent regression

### For AI Agents

Follow the workflow in `AGENTS.md`:
1. ✅ Review specs first
2. ✅ Propose spec updates
3. ✅ Wait for approval
4. ✅ Implement 100% according to specs
5. ✅ Test against spec acceptance criteria

### For Contributors

See [CONTRIBUTING.md](../CONTRIBUTING.md#spec-driven-development-workflow) for spec contribution guidelines.

---

## Spec Conventions

### Naming

- **Product specs**: `feature-name.md` (descriptive, lowercase-hyphenated)
- **RFCs**: `NNNN-short-description.md` (numbered sequentially)
- **API specs**: `api-name.md` (descriptive)
- **Testing specs**: `test-type.feature` (BDD feature file format)

### Versioning

All specs include:
- Version number (e.g., v1.0.0)
- Status (Draft, Review, Approved, Implemented)
- Last updated date

### Traceability

- Product requirements use `REQ-N` IDs
- RFCs reference requirement IDs in task checklists
- Test specs map properties (P1-P16) to requirements

---

## Related Documents

| Document | Location |
|----------|----------|
| AI Agent Workflow | [`AGENTS.md`](../AGENTS.md) |
| Contributing Guide | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| CLAUDE Configuration | [`CLAUDE.md`](../CLAUDE.md) |
| User Documentation | [`docs/`](../docs/) |

---

**Last Updated**: 2025-04-16
