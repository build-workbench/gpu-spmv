# 贡献指南

感谢你对 GPU SpMV 的贡献兴趣！

## 开发环境设置

### 前置要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 编译器
- Git

### 克隆和构建

```bash
git clone https://github.com/LessUp/gpu-spmv.git
cd gpu-spmv
cmake --preset default
cmake --build --preset default
```

## Spec-Driven 工作流程

GPU SpMV 遵循 **OpenSpec** 规范驱动开发：

1. **阅读规范** `openspec/specs/<功能>/spec.md`
2. **更新规范** 如需更改（需讨论）
3. **实现** 按规范执行
4. **测试** 验证规范要求
5. **文档** 记录设计决策

## 代码风格

- 4 空格缩进
- 100 字符行宽
- Google C++ 风格
- 使用 `clang-format`（版本 18）

```bash
find src include tests -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) | xargs clang-format -i
```

## 提交规范

```
feat(scope): 描述    # 新功能
fix(scope): 描述     # Bug 修复
perf(scope): 描述    # 性能优化
refactor(scope): 描述 # 重构
docs(scope): 描述    # 文档
test(scope): 描述    # 测试
```

## Pull Request 流程

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 运行测试：`ctest --preset default`
5. 格式化代码：`clang-format`
6. 提交 PR 并附描述

## 文档

### 构建文档

```bash
cd docs
npm install
npm run dev
```

### 添加页面

- 中文文档：`docs/zh/`
- 英文文档：`docs/en/`
- 使用 Mermaid 绘制图表

## 获取帮助

- 提交 [Issue](https://github.com/LessUp/gpu-spmv/issues)
- 查看现有文档
- 阅读 OpenSpec 规范

## 许可证

贡献即表示你同意你的贡献将按 MIT 许可证授权。
