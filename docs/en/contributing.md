# Contributing

Thank you for your interest in contributing to GPU SpMV!

## Development Setup

### Prerequisites

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- Git

### Clone and Build

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset default
cmake --build --preset default
```

## Spec-Driven Workflow

GPU SpMV follows **OpenSpec** specification-driven development:

1. **Read the spec** in `openspec/specs/<feature>/spec.md`
2. **Update spec** if changes are needed (requires discussion)
3. **Implement** according to the spec
4. **Test** against spec requirements
5. **Document** any design decisions

## Code Style

- 4-space indentation
- 100-character line width
- Google C++ style guide
- Use `clang-format` (version 18)

```bash
find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i
```

## Commit Convention

```
feat(scope): description    # New feature
fix(scope): description     # Bug fix
perf(scope): description    # Performance optimization
refactor(scope): description # Refactoring
docs(scope): description    # Documentation
test(scope): description    # Testing
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `ctest --preset default`
5. Format code: `clang-format`
6. Submit PR with description

## Documentation

### Building Docs

```bash
cd docs
npm install
npm run dev
```

### Adding Pages

- Chinese docs: `docs/zh/`
- English docs: `docs/en/`
- Use Mermaid for diagrams

## Getting Help

- Open an [Issue](https://github.com/LessUp/gpu-spmv/issues)
- Check existing documentation
- Review OpenSpec specs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
