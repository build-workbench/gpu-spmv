#CLAUDE.md — Claude Code 专属配置

> Claude Code(claude.ai / code) 在本仓库工作时的专项指南。
    > 通用项目规范见 `AGENTS.md`，本文件仅描述 Claude 特有行为。

          ##语言要求

              **始终使用中文回复用户 **，代码注释保持英文。

          -- -

          ##Claude 特有行为

              ## #Hook 配置

              项目已配置 `PostToolUse` hook，在写入
              / 编辑 C++ / CUDA 文件后自动执行 `clang
          - format -
          i`。

          ## #OpenSpec 命令

          Claude 可通过以下命令管理变更提案：
          - `/ opsx : propose` — 创建变更提案 - `/ opsx : apply` — 实现当前提案任务 - `/ opsx
    : archive` — 归档已完成变更

      ## #强制工作流

      阅读 spec → 更新 spec（必要时）→ 用户确认 → 实现 → 测试

      -- -

          ##CI 特殊说明

          - CI 无 GPU：`benchmarks / main.cu`、`src / pagerank.cu` 无 GPU 时自动退出 - CI 使用 clang
          - format - 18 检查格式 - CPU - only 构建：`cmake - S.- B build - no - cuda
          - DSPMV_REQUIRE_CUDA = OFF`
