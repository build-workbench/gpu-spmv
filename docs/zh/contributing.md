# 贡献指南

感谢你关注 GPU SpMV。

## 开发环境

```bash
git clone https://github.com/AICL-Lab/gpu-spmv.git
cd gpu-spmv
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

无 GPU 环境可使用：

```bash
cmake -S . -B build-no-cuda -DSPMV_REQUIRE_CUDA=OFF
cmake --build build-no-cuda
ctest --test-dir build-no-cuda --output-on-failure
```

Linux 下请优先使用官方 CUDA preset，让构建固定走系统 GCC/G++ host toolchain：

```bash
cmake --preset cuda-linux
cmake --build --preset cuda-linux
ctest --preset cuda-linux
```

Release 构建可使用：

```bash
cmake --preset cuda-linux-release
cmake --build --preset cuda-linux-release
ctest --preset cuda-linux-release
```

## 贡献规则

1. 变更应聚焦核心 SpMV 库。
2. 保持 RAII 资源管理和显式错误处理。
3. 使用现有测试命令完成验证。
4. 行为变化时同步更新相关文档。

## 代码风格

- 4 空格缩进
- 100 字符行宽
- Google C++ 风格
- 修改过的文件使用 `clang-format`

## 文档

- 中文文档位于 `docs/zh/`
- 可使用 Mermaid 绘图

## 获取帮助

- 提交 [Issue](https://github.com/AICL-Lab/gpu-spmv/issues)
- 阅读现有文档
