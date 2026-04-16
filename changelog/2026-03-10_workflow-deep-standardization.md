# Workflow 深度标准化

**日期**: 2026-03-10  
**关联 PR**: #4, #5  
**类型**: 基础设施

---

## 🔄 变更 (Changed)

### CI Workflow 标准化
- 统一 `permissions: contents: read` 与 `concurrency` 配置

### Pages Workflow 改进
- 补充 `actions/configure-pages@v5` 步骤
- 添加 `paths` 触发过滤，减少无效构建

## ✨ 新增 (Added)

### 安全性增强
- 显式声明 workflow 权限，遵循最小权限原则

### 并发控制
- 添加 `concurrency` 配置，避免重复构建

---

## 背景

全仓库第二轮 GitHub Actions 深度标准化：
- 统一命名、权限、并发、路径过滤与缓存策略
- 确保所有 workflow 遵循 GitHub 最佳实践

---

## 变更详情

| Workflow | 变更 |
|:---------|:-----|
| `ci.yml` | 添加权限声明、并发控制 |
| `pages.yml` | 添加 configure-pages 步骤、路径过滤 |

---

## 检查清单

- [x] 所有 workflow 语法检查通过
- [x] Pages 部署验证成功
- [x] 权限配置符合安全最佳实践
