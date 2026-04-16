# Changelog Directory

本目录记录 GPU SpMV 项目的详细变更历史。

## 文件说明

| 文件 | 描述 |
|:-----|:-----|
| `CHANGELOG.md` | 主更新日志，汇总所有版本变更 |
| `template.md` | 变更记录模板 |
| `YYYY-MM-DD_*.md` | 详细的变更记录 |

## 变更记录规范

### 文件命名格式

```
YYYY-MM-DD_brief-description.md
```

示例:
- `2026-03-22_phase1-execution-path-optimization.md`
- `2026-03-13_workflow-cpu-safe-ci.md`

### 内容格式

遵循 [Keep a Changelog](https://keepachangelog.com/) 标准格式：

```markdown
# 标题

日期: YYYY-MM-DD

## 分类

### Added / 新增
- 新功能描述

### Changed / 变更
- 功能变更描述

### Fixed / 修复
- Bug 修复描述

## 背景

变更原因说明

## 影响

- 用户影响
- 性能影响
- 兼容性影响
```

## 分类说明

| 分类 | 含义 | 版本 bump |
|:-----|:-----|:---------:|
| Added | 新功能 | Minor |
| Changed | 功能变更 | Minor |
| Deprecated | 即将移除 | Minor |
| Removed | 移除功能 | Major |
| Fixed | Bug 修复 | Patch |
| Security | 安全修复 | Patch |

## 链接

- [主更新日志](../CHANGELOG.md)
- [语义化版本规范](https://semver.org/lang/zh-CN/)
