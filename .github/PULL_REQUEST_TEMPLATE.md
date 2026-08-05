## 描述

<!-- 简要描述改动 -->

## 改动类型

- [ ] 🐛 Bug 修复（不破坏现有功能的修复）
- [ ] ✨ 新功能（不破坏现有功能的新增）
- [ ] 💥 破坏性变更（会导致现有功能失效的修复或功能）
- [ ] 📚 文档更新
- [ ] 🔧 构建/CI 改进
- [ ] ♻️ 重构（无功能变化）

## 测试

- [ ] 相关测试全部通过（Linux CUDA 用 `ctest --preset cuda-linux`，CPU-only 用 `ctest --preset cpu-only`）
- [ ] 为新功能添加了新测试
- [ ] 代码已格式化：`find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i`

## 代码质量

- [ ] 代码遵循风格规范（4 空格缩进，100 字符行宽）
- [ ] 没有裸 `cudaMalloc`/`cudaFree`（使用 `CudaBuffer<T>`）
- [ ] 使用 `SpMVError` 枚举正确处理错误
- [ ] include 顺序：`"spmv/"` → `<cuda*>` → `<标准库>` → `<第三方>`

## 文档

- [ ] 更新了 README.md（如适用）
- [ ] 更新了 docs/（如有用户可见变化）

## 补充说明

<!-- 给 reviewer 的额外信息 -->
