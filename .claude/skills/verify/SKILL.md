---
name: verify
description: Configure, build, and run tests for the SpMV project. Uses CPU-only path when no GPU is available.
---

Run the full verification pipeline for this project:

1. **Configure** the project:
   ```bash
   cmake --preset default
   ```

2. **Build**:
   ```bash
   cmake --build --preset default -j$(nproc)
   ```

3. **Run tests**:
   ```bash
   ctest --preset default --output-on-failure
   ```

If any step fails, report the error clearly with the relevant output. Do not attempt to fix issues automatically — report them and wait for instructions.

Note: Tests requiring a CUDA device will be skipped or fail if no GPU is available. This is expected in CI and headless environments.
