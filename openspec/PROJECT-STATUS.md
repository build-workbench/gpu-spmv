#GPU SpMV 项目状态报告

> **生成日期** : 2026 - 04 - 28 > **项目版本** : v1 .0.0 >
            **状态** : ✅ 已归档

    -- -

                ##项目概述

                    ** GPU SpMV** 是基于 CUDA
                        的高性能稀疏矩阵向量乘法库，支持 4 种优化内核和自动内核选择。

                ## #核心特性 -
                4 种 CUDA Kernel（Scalar CSR / Vector CSR / Merge Path / ELL） -
                2 种稀疏格式（CSR / ELL） - 自动内核选择（基于 avg_nnz 和 skewness） -
                70 % +带宽利用率 -
                PageRank 算法示例

                -- -

                ##已完成功能

        | 功能 | 规范路径 | 完成日期 | | -- -- -- | -- -- -- -- -- | -- -- -- -- -- | |
        CSR 格式 | `openspec / specs / csr - format /` | 2025 - 01 - 15 | |
        ELL 格式 | `openspec / specs / ell - format /` | 2025 - 02 - 10 | |
        SpMV 内核 | `openspec / specs / spmv - kernels /` | 2025 - 02 - 20 | |
        内核选择器 | `openspec / specs / spmv - kernels /` | 2025 - 03 - 01 | |
        基准测试 | `openspec / specs / benchmark /` | 2025 - 03 - 05 | |
        PageRank | `openspec / specs / pagerank /` | 2025 - 03 - 10 | |
        项目收尾 | `openspec / changes / archive / 2026 - 04 - project - completion /` |
        2026 - 04 - 28 |

        -- -

            ##收尾完成项

            ## #OpenSpec 规范修复 -
            [x] 修复 `public - api / spec.md` 中 API 签名不一致 - [x] 删除 `csr -
            format / spec.md` 中无效 RFC 引用 -
            [x] 添加 `SpMVThresholds` 相关 API 文档

                ## #CI /
                CD 优化 -
            [x] 修复 clang -
            tidy 正确报告错误级别

            ## #文档治理 -
            [x] 删除重复的 `docs / changelog.md` 和 `docs / changelog.en.md` -
            [x] 更新 GitHub 仓库描述和 topics

            ## #工程化完善 -
            [x] 归档 `project - completion` 提案 - [x] 清理所有构建目录 -
            [x] 更新 `.gitignore` 排除运行时状态

            -- -

            ##最终 Git 提交

``` 1e2c232 chore : 添加.omc 运行时状态到 gitignore 6296cc4 refactor(openspec)
    : 规范化重构与项目收尾
```

      -- -

      ##仓库信息

            - **GitHub *
                  * : https
    :  // github.com/LessUp/gpu-spmv
       -**文档 *
                  * : https
    :  // lessup.github.io/gpu-spmv/
       -**License *
                  * : MIT

                      -- -

            ##后续维护说明

                本项目已达到稳定状态，核心功能完整且经过验证。如需继续开发：

                1. 参考 `AGENTS.md` 了解项目规范 2. 使用 OpenSpec 工作流管理变更 3. 查看 `openspec
                / specs /` 了解功能规范 4. 运行 `cmake-- preset default &&
    ctest-- preset default` 验证

        -- -

        *此报告由 AI 自动生成，标记项目收尾完成。*
