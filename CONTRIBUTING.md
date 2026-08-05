# Contributing to GPU SpMV

Keep contributions narrow, verifiable, and centered on the core SpMV library.

## Development setup

```bash
git clone https://github.com/AICL-Lab/gpu-spmv.git
cd gpu-spmv

cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

CPU-only environments:

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda
ctest --test-dir build-no-cuda --output-on-failure
```

On Linux, use the dedicated CUDA presets so the build always uses the system GCC/G++ toolchain
instead of Conda host compilers:

```bash
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

Release builds:

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## GPU CI

Kernel correctness tests (`tests/test_spmv.cu`, `tests/test_bandwidth.cu`) need a real
NVIDIA GPU, which GitHub's hosted runners do not provide. CI therefore has two CUDA
gates:

- `cuda-compile` — always runs; compiles the library and tests in an NVIDIA container.
- `cuda-test` — runs the full test suite on a self-hosted GPU runner. It is skipped
  unless the repository variable `CUDA_RUNNER_LABEL` is set (Settings → Secrets and
  variables → Actions → Variables). Set it to the label of a self-hosted runner that
  has the CUDA toolkit, CMake, and a C++ compiler on `PATH`.

Before merging kernel changes, run the suite locally on a GPU machine:

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## Benchmarks and examples

- `examples/basic_spmv.cpp` builds by default (`SPMV_BUILD_EXAMPLES=ON`) and runs in
  both CUDA and CPU-only builds.
- The synthetic benchmark tool builds with `-DSPMV_BUILD_BENCHMARKS=ON` (CUDA only)
  and can also load Matrix Market files: `./build/spmv_bench matrix.mtx`.

## What belongs in this repository

Good contributions:

- Improve CSR / ELL storage or validation
- Improve kernel selection or execution reliability
- Fix correctness, memory-safety, or error-reporting issues
- Simplify documentation for the core library

Bad contributions:

- New AI governance layers or repository-specific agent workflows
- Showcase modules that are not part of the core SpMV library
- Large process frameworks that add more maintenance than value

## Code guidelines

- Use C++17
- Keep 4-space indentation and 100-character lines
- Prefer existing helpers and explicit error handling
- Do not introduce raw `cudaMalloc` / `cudaFree`; use `CudaBuffer<T>`
- Keep include order: project → CUDA → standard library → third party

Format changed files with:

```bash
find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) \
  | xargs clang-format -i
```

## Before opening a pull request

1. Run the relevant build and test commands.
2. Update README / docs when user-visible behavior changes.
3. Keep the change focused; avoid bundling unrelated cleanup.
4. Record project-level changes in the root `CHANGELOG.md` when needed.

## Commit messages

Use Conventional Commits:

```text
feat(scope): description
fix(scope): description
refactor(scope): description
docs(scope): description
test(scope): description
```
