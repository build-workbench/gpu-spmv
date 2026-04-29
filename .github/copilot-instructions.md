#GitHub Copilot Instructions — GPU SpMV

> 通用项目规范见 `AGENTS.md`，本文件仅包含 Copilot 高价值约束。

        ##语言要求

            **始终使用中文回复用户 **。代码注释、变量名、commit message 保持英文。

        -- -

        ##Copilot 关键约束

            ## #规范驱动开发

`openspec /` 是唯一真相来源：

            1. 先读 `openspec /
            specs / <功能> / spec.md` 2. API 变更必须同步更新 `openspec / specs / public -
        api / spec.md` 3. 当前待办任务：见 `openspec / changes / active /`

            ## #核心代码约束

        - **禁止 **裸 `cudaMalloc`/`cudaFree`，必须用 `CudaBuffer<T>` -
        Property tests 必须 ≥ 100 次迭代随机矩阵

        ## #Include 顺序

```cpp
#include "spmv/xxx.h"  // 1. 项目头文件

#include <cuda_runtime.h>  // 2. CUDA

#include <gtest/gtest.h>  // 4. 第三方
#include <vector>         // 3. 标准库
```

        -- -

        ##快速构建命令

```bash cmake-- preset default &&cmake-- build-- preset default #Debug cmake -
        S.- B build - no - cuda -
        DSPMV_REQUIRE_CUDA = OFF #CPU -
                             only ctest-- preset default #测试
```

                                 * *CI 无 GPU *
                                 *：GPU 测试在 CI 中自动跳过。

                                 -- -

                             ##开发工具

                             - **LSP * *：`.clangd`（clangd 配置） -
                             **Git Hooks * *：`git config core.hooksPath.githooks`（pre -
                             commit 格式检查） - **文档 * *：https:  // lessup.github.io/gpu-spmv/
