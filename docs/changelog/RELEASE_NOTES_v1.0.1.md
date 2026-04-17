# GPU SpMV v1.0.1 Release Notes

## 🎉 发布标题 / Release Title

**English**: GPU SpMV v1.0.1 - Documentation Overhaul with Bilingual Support

**中文**: GPU SpMV v1.0.1 - 文档全面重构与双语支持

---

## 📝 发布说明 / Release Notes

### English

This release focuses on a comprehensive documentation overhaul with full bilingual (English/Chinese) support.

#### ✨ What's New

**Complete Bilingual Documentation**
- All documentation now available in both English and Chinese
- Language switcher on every page
- Consistent structure and quality across languages

**New Documentation**
- 📦 **Installation Guide** - Platform-specific instructions for Ubuntu, CentOS, Windows, macOS
- 🏗️ **Architecture Design** - System architecture, core algorithms, design decisions  
- 🚀 **Performance Guide** - Optimization strategies, benchmarking, best practices

**Refactored Documentation**
- 📚 **API Reference** - Clearer structure with examples
- 📝 **Examples** - More concise and practical code samples
- 📋 **Changelog** - Standardized Keep a Changelog format

#### 📚 Documentation Structure

```
docs/
├── _config.yml              # Bilingual navigation
├── index.md / index.en.md   # Homepage
├── installation.md/.en.md   # Installation guides
├── architecture.md/.en.md   # Architecture design
├── api.md/.en.md            # API reference
├── examples.md/.en.md       # Code examples
├── performance.md/.en.md    # Performance optimization
└── changelog.md/.en.md      # Version history
```

#### 📊 Stats
- 23 files changed
- 4,079 insertions(+), 1,760 deletions(-)
- 100% bilingual documentation coverage

---

### 简体中文

本次发布专注于全面的文档重构，提供完整的中英文双语支持。

#### ✨ 新增内容

**完整的中英文双语文档**
- 所有文档均提供中英文两个版本
- 每页都有语言切换链接
- 跨语言保持一致的结构和质量

**新增文档**
- 📦 **安装指南** - 针对 Ubuntu、CentOS、Windows、macOS 的详细说明
- 🏗️ **架构设计** - 系统架构、核心算法、设计决策
- 🚀 **性能优化指南** - 优化策略、基准测试、最佳实践

**重构的文档**
- 📚 **API 参考** - 更清晰的结构，附带示例
- 📝 **示例代码** - 更简洁实用的代码样本
- 📋 **更新日志** - 标准化的 Keep a Changelog 格式

#### 📚 文档结构

```
docs/
├── _config.yml              # 双语导航
├── index.md / index.en.md   # 首页
├── installation.md/.en.md   # 安装指南
├── architecture.md/.en.md   # 架构设计
├── api.md/.en.md            # API 参考
├── examples.md/.en.md       # 示例代码
├── performance.md/.en.md    # 性能优化
└── changelog.md/.en.md      # 更新日志
```

#### 📊 统计
- 23 个文件变更
- 4,079 行新增, 1,760 行删除
- 100% 双语文档覆盖率

---

## 🌐 Documentation Site

**URL**: https://lessup.github.io/gpu-spmv/

---

## 🏷️ Git Tag

Tag: `v1.0.1`

Commit: `f4c0820`

---

## 📋 Manual Release Command

Since gh CLI requires authentication, you can manually create this release using:

### Method 1: Web Interface
1. Go to: https://github.com/LessUp/gpu-spmv/releases/new
2. Choose tag: `v1.0.1` (create new if needed)
3. Enter title: `GPU SpMV v1.0.1 - Documentation Overhaul with Bilingual Support`
4. Copy contents from this file to the release notes
5. Click "Publish release"

### Method 2: gh CLI (if authenticated)
```bash
gh release create v1.0.1 \
  --title "GPU SpMV v1.0.1 - Documentation Overhaul with Bilingual Support" \
  --notes-file RELEASE_NOTES_v1.0.1.md
```

---

## 📄 License

MIT License
