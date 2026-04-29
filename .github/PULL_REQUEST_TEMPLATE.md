## Description

<!-- Brief description of changes -->

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Build/CI improvement
- [ ] ♻️ Refactoring (no functional changes)

## Spec Updates

This project follows **Spec-Driven Development**. If this PR modifies behavior:

- [ ] Updated `openspec/specs/<feature>/spec.md`
- [ ] Updated `openspec/specs/public-api/spec.md` (if API changed)
- [ ] Created proposal in `openspec/changes/active/` (for new features)

## Testing

- [ ] All tests pass: `ctest --preset default`
- [ ] Added new tests for new functionality
- [ ] Property tests run with ≥ 100 iterations
- [ ] Code formatted: `find src include tests benchmarks -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i`

## Code Quality

- [ ] Code follows style guidelines (4-space indent, 100-char line limit)
- [ ] No naked `cudaMalloc`/`cudaFree` (use `CudaBuffer<T>`)
- [ ] Proper error handling with `SpMVError` enum
- [ ] Include order: `"spmv/"` → `<cuda*>` → `<standard>` → `<third-party>`

## Documentation

- [ ] Updated README.md and/or README.zh-CN.md (if applicable)
- [ ] Updated docs/ (if user-facing change)
- [ ] Updated CHANGELOG.md

## Additional Notes

<!-- Any additional information for reviewers -->
