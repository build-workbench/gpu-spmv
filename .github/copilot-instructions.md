#GitHub Copilot Instructions — GPU SpMV

> **Copilot 专属约束* *
            -完整项目规范见 `AGENTS.md`

             -- -

             ##核心约束(MUST)

                 1. *
            *语言** : 中文回复，代码注释 / commit 保持英文 2. * *规范驱动** : `openspec / specs
            /` 是唯一真相来源，先读 spec 再编码 3. *
            *内存安全** : 禁止裸 `cudaMalloc`/`cudaFree`，用 `CudaBuffer<T>`

            -- -

            ##代码规范速查

            ## #Include 顺序
```cpp
#include "spmv/xxx.h"  // 1. 项目头文件

#include <cuda_runtime.h>  // 2. CUDA

#include <gtest/gtest.h>  // 4. 第三方
#include <vector>         // 3. 标准库
```

            ## #命名约定
    | 类别 | 风格 | 示例 | | -- -- -- | -- -- -- | -- -- -- | | 类型 | PascalCase | `CSRMatrix` |
    | 函数 | snake_case | `csr_create` | | 常量 | UPPER_SNAKE_CASE | `WARP_SIZE` |

    ## #格式 - 4 空格缩进，100 字符行宽 -
        Property tests ≥ 100 次迭代

        -- -

        ##快速命令

```bash
#构建
        cmake-- preset default&& cmake-- build-- preset default

#CPU - only(无 GPU)
        cmake
        - S.- B build - no - cuda - DSPMV_REQUIRE_CUDA = OFF && cmake-- build build - no -
                                                                        cuda

#测试
                                                                        ctest-- preset default

#格式化
                                                                        find src include tests
                                                                        - name "*.cpp" - o
                                                                        - name "*.h" |
                                                                    xargs clang - format -
                                                                        i
```

                                                                        -- -

                                                                        ##更多信息

                                                                        - **完整规范 * *
    : `AGENTS.md` - **API 规范 * * : `openspec / specs / public - api / spec.md` -
                                                                        **在线文档 * * : https
    :  // lessup.github.io/gpu-spmv/
