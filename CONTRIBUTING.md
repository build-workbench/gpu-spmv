#Contributing to GPU SpMV

Thank you for your interest in contributing to GPU SpMV! This guide will help you get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Spec-Driven Development Workflow](#spec-driven-development-workflow)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (OS, CUDA version, GPU model)
- **Code examples** if applicable

### Suggesting Enhancements

Enhancement suggestions should:

- Explain the **problem** the enhancement solves
- Describe the **proposed solution**
- Include **use cases** and examples
- Reference relevant **specification documents** (if any)

### Your First Code Contribution

Unsure where to start? Look for issues labeled:

- `good first issue` - Perfect for newcomers
- `help wanted` - Issues where maintainers need assistance
- `docs` - Documentation improvements

---

## Spec-Driven Development Workflow

**IMPORTANT**: This project follows **Spec-Driven Development (SDD)**. All contributions must adhere to our spec-first workflow.

### What is SDD?

In SDD, specification documents in `openspec/specs/` are the **Single Source of Truth**. Code implementation follows specs, not the other way around.

### Spec Directory Structure

```
openspec/
├── config.yaml              # Project configuration
├── specs/                   # Feature specifications (single source of truth)
│   ├── csr-format/          # CSR format spec + design
│   ├── ell-format/          # ELL format spec + design
│   ├── spmv-kernels/        # Kernel implementations
│   ├── public-api/          # Public API specification (update on any API change)
│   ├── error-handling/      # Error handling spec
│   ├── benchmark/           # Benchmark spec
│   ├── pagerank/            # PageRank algorithm spec
│   └── property-tests/      # Test requirements
└── changes/
    ├── active/              # Current iteration tasks
    └── archive/             # Completed changes
```

### Contributing to Specs

#### When to Update Specs

1. **New features**: Create new spec in `openspec/specs/`
2. **API changes**: Update `openspec/specs/public-api/spec.md` before code changes
3. **Architecture changes**: Create design document in `openspec/specs/<feature>/design.md`
4. **Test coverage gaps**: Update `openspec/specs/property-tests/spec.md`

#### Spec Update Process

1. **Identify relevant specs**: Check which spec files need updates
2. **Create proposal**: Update spec documents with clear rationale
3. **Get review**: Discuss changes in PR comments
4. **Implement code**: After spec approval, implement according to specs
5. **Verify**: Ensure code meets spec acceptance criteria

#### Spec File Naming

- **Feature specs**: `openspec/specs/<feature>/spec.md` (e.g., `openspec/specs/csr-format/spec.md`)
- **Design docs**: `openspec/specs/<feature>/design.md` (technical decisions)
- **API spec**: `openspec/specs/public-api/spec.md` (all public API)
- **Test spec**: `openspec/specs/property-tests/spec.md`

### AI Agent Workflow

If you're using AI coding assistants (Claude, Cursor, etc.), they MUST follow:

1. **Review specs first** before writing code
2. **Propose spec updates** for new functionality
3. **Wait for approval** on specs before implementation
4. **Implement 100% according to specs**
5. **Test against spec acceptance criteria**

See `AGENTS.md` for detailed AI workflow instructions.

---

## Development Setup

### Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Standard | C++17 | C++17 |
| NVIDIA GPU | CC 7.0 (Volta) | CC 8.6+ (Ampere) |

### Quick Start

```bash
#Clone repository
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv

#Build(Debug mode for development)
cmake --preset default
cmake --build --preset default

#Run tests
ctest --preset default
```

### Build Commands

```bash
#Debug build(with symbols, no optimization)
cmake --preset default && cmake --build --preset default

#Release build(optimized)
cmake --preset release && cmake --build --preset release

#CPU - only build(no CUDA device required)
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda

#Run specific tests
./build/spmv_tests --gtest_filter="CSR*"
```

### Code Formatting

```bash
#Format all source files
find src tests include -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
```

---

## Pull Request Process

### Before Submitting

1. **Update specs first** (if adding/modifying features)
2. **Ensure tests pass**: `ctest --preset default`
3. **Format code**: Run clang-format
4. **Update documentation**: README, CHANGELOG, API docs
5. **Squash commits**: Use clean commit history

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Spec Updates

- [ ] Updated `openspec/specs/<feature>/spec.md`
- [ ] Updated `openspec/specs/public-api/spec.md` (if API changed)
- [ ] Created proposal in `openspec/changes/active/` (for new features)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added/updated unit tests
- [ ] Added/updated property tests (100 iterations)
- [ ] All tests pass: `ctest --preset default`

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Code is formatted with clang-format
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

### Review Process

1. **Spec review**: Ensure specs are updated and complete
2. **Code review**: Verify implementation matches specs
3. **Test review**: Check test coverage and property tests
4. **Merge**: After approval and CI passes

---

## Style Guidelines

### C++ Style

- **Formatting**: Google style via clang-format
  - 4-space indentation
  - 100 character line limit
  - Braces on same line

- **Naming conventions**:
  - Types: `PascalCase` (e.g., `CSRMatrix`, `SpMVConfig`)
  - Functions: `snake_case` (e.g., `csr_create`, `spmv_csr`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_BLOCK_SIZE`)
  - Private members: `snake_case_` suffix (e.g., `ptr_`, `size_`)

- **Include order**:
  1. Project headers: `"spmv/..."`
  2. CUDA headers: `<cuda_runtime.h>`, etc.
  3. Standard library: `<vector>`, `<string>`, etc.
  4. Third-party: `<gtest/gtest.h>`, etc.

- **Namespace**: All code in `spmv::` namespace

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:

| Type | Use When |
|------|----------|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `build` | Build system changes |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `docs` | Documentation changes |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |

**Examples**:

```
feat(spmv): add merge path kernel for load balancing
fix(csr): correct element lookup for empty rows
perf(ell): optimize column-major access pattern
build(cmake): add minimal preset for sm_80
docs(api): update public API specification
```

---

## Testing Requirements

### Test Types

1. **Unit Tests**: Test specific functionality
2. **Property Tests**: Validate general properties with random data (100 iterations minimum)
3. **Performance Tests**: Measure execution time and bandwidth

### Writing Property Tests

Property tests must run at least **100 iterations** with randomly generated matrices:

```cpp
TEST(SpMVPropertyTest, MyNewProperty) {
    for (int iter = 0; iter < 100; iter++) {
        // Generate random test data
        auto matrix = generate_random_sparse_matrix();
        auto x = generate_random_vector(matrix->num_cols);

        // Execute and validate
        auto result = spmv_csr(matrix, d_x, d_y);

        // Assert property holds
        EXPECT_TRUE(property_valid(result));
    }
}
```

### Test Coverage

Target **>80% coverage** for core functionality. Validate:

- ✅ Correctness vs CPU reference implementation
- ✅ Edge cases (empty matrices, dimension mismatches)
- ✅ Error handling (invalid inputs, memory failures)
- ✅ Performance metrics (bandwidth, GFLOPS)

---

## Documentation

### Spec Documentation

Keep `openspec/specs/` directory synchronized with code:

- **Feature specs**: Update when requirements change
- **Design docs**: Document major architectural decisions
- **API spec**: Update with every API change
- **Test spec**: Document all property tests

### User Documentation

Located in `/docs/` and rendered via GitHub Pages:

- **Installation guides**: Setup instructions
- **Tutorials**: Step-by-step examples
- **API reference**: Auto-generated from headers
- **Architecture docs**: High-level design overview

### README Updates

Update `README.md` (English) and `README.zh-CN.md` (Chinese) when:

- Adding new features
- Changing quick start examples
- Updating performance benchmarks
- Modifying project structure

---

## Questions?

- **Technical questions**: Open a GitHub Discussion
- **Spec clarifications**: Comment on relevant spec files
- **Bug reports**: Create GitHub Issue with reproduction steps

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GPU SpMV! 🎉
