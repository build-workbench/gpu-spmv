## Description

<!-- Brief description of changes -->

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Build/CI improvement
- [ ] ♻️ Refactoring (no functional changes)

## Testing

- [ ] All relevant tests pass (`ctest --preset cuda-linux` on Linux CUDA, or `ctest --test-dir build-no-cuda --output-on-failure` for CPU-only)
- [ ] Added new tests for new functionality
- [ ] Code formatted: `find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i`

## Code Quality

- [ ] Code follows style guidelines (4-space indent, 100-char line limit)
- [ ] No naked `cudaMalloc`/`cudaFree` (use `CudaBuffer<T>`)
- [ ] Proper error handling with `SpMVError` enum
- [ ] Include order: `"spmv/"` → `<cuda*>` → `<standard>` → `<third-party>`

## Documentation

- [ ] Updated README.md (if applicable)
- [ ] Updated docs/ (if user-facing change)

## Additional Notes

<!-- Any additional information for reviewers -->
