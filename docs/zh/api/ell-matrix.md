# ELL 矩阵

ELL (ELLPACK) 稀疏矩阵数据结构和操作。

## 数据结构

```cpp
struct ELLMatrix {
    int num_rows;         // 行数
    int num_cols;         // 列数
    int max_nnz_per_row;  // 每行最大非零元素数（决定填充）
    int nnz;              // 实际非零元素总数

    // 列主序存储: values[k * num_rows + row]
    float* values;     // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;  // 列索引数组，-1 表示填充

    void* internal;    // 不透明内部状态（设备内存管理）
};
```

## 核心函数

### 创建与销毁

```cpp
ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row);
void ell_destroy(ELLMatrix* mat);
```

### 数据转换

```cpp
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);
int ell_to_dense(const ELLMatrix* ell, float* dense);
```

### GPU 数据传输

```cpp
int ell_to_gpu(ELLMatrix* mat);
int ell_from_gpu(ELLMatrix* mat);
```

### 元素访问

```cpp
float ell_get_element(const ELLMatrix* mat, int row, int col);
```

## 内存布局

```
原始矩阵:           ELL 存储（列优先）:
┌─────┬─────┐       values:     [ 1, 3, 5,   ← 第 0 列
│ 1 0 2 │                     2, 4, 0 ]  ← 第 1 列
│ 0 3 4 │   =>  col_indices: [ 0, 1, 3,   ← 第 0 列
│ 0 0 5 │                     2, 2, - ]  ← 第 1 列 (-1 = 填充)
└─────┴─────┘       
                    每行补齐到最大长度
```

**特点**:
- 列优先存储，GPU 合并访存
- 无需行指针，减少内存访问
- 对齐填充，适合 SIMD 执行
- 行长度均匀时性能最佳

## 何时使用 ELL

| 条件 | 建议 |
|:-----|:-----|
| 行长度均匀（倾斜度 < 3） | 使用 ELL |
| 行长度差异大 | 使用 CSR |
| 需要最大 GPU 性能 | 转换为 ELL |

## 示例

```cpp
#include <spmv/spmv.h>

int main() {
    // 先创建 CSR 矩阵
    CSRMatrix* csr = csr_create(1000, 1000, 20000);
    // ... 填充数据 ...

    // 检查是否值得转换为 ELL
    CSRStats stats = csr_compute_stats(csr);
    if (stats.skewness < 3.0f) {
        // 转换为 ELL
        ELLMatrix* ell = ell_create(csr->num_rows, csr->num_cols,
                                    stats.max_nnz_per_row);
        ell_from_csr(ell, csr);
        ell_to_gpu(ell);

        // 执行 ELL SpMV
        SpMVResult result = spmv_ell(ell, d_x, d_y, nullptr);

        ell_destroy(ell);
    }

    csr_destroy(csr);
    return 0;
}
```
