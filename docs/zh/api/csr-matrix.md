# CSR 矩阵

CSR (Compressed Sparse Row) 稀疏矩阵数据结构和操作。

## 数据结构

```cpp
struct CSRMatrix {
    int num_rows;  // 行数
    int num_cols;  // 列数
    int nnz;       // 非零元素总数

    float* values;     // 非零值数组 [nnz]
    int* col_indices;  // 列索引数组 [nnz]
    int* row_ptrs;     // 行指针数组 [num_rows + 1]

    // GPU 设备指针
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;

    // 内存所有权标志
    bool owns_host_memory;
    bool owns_device_memory;
};
```

## 核心函数

### 创建与销毁

```cpp
CSRMatrix* csr_create(int num_rows, int num_cols, int nnz);
void csr_destroy(CSRMatrix* mat);
```

### 数据转换

```cpp
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);
int csr_to_dense(const CSRMatrix* csr, float* dense);
```

### GPU 数据传输

```cpp
int csr_to_gpu(CSRMatrix* mat);
int csr_from_gpu(CSRMatrix* mat);
void csr_free_gpu(CSRMatrix* mat);
```

### 元素访问

```cpp
float csr_get_element(const CSRMatrix* mat, int row, int col);
```

### 序列化

```cpp
int csr_serialize(const CSRMatrix* mat, const char* filename);
int csr_deserialize(CSRMatrix* mat, const char* filename);
```

### 统计与验证

```cpp
CSRStats csr_compute_stats(const CSRMatrix* mat);
bool csr_validate(const CSRMatrix* mat);
```

## CSRStats 结构

```cpp
struct CSRStats {
    float avg_nnz_per_row;  // 平均每行非零元素数
    int max_nnz_per_row;    // 最大每行非零元素数
    int min_nnz_per_row;    // 最小每行非零元素数
    float skewness;         // 倾斜度: max / (min + 1)
};
```

## 内存布局

```
原始矩阵:           CSR 存储:
┌─────┬─────┐       values:     [ 1, 2, 3, 4, 5 ]
│ 1 0 2 │       col_indices: [ 0, 2, 1, 2, 3 ]
│ 0 3 4 │   =>  row_ptrs:    [ 0, 2, 4, 5 ]
│ 0 0 5 │       
└─────┴─────┘       row_ptrs[i] 表示第 i 行的起始位置
```

**特点**:
- 通用格式，存储高效
- 适合不规则稀疏模式
- 行遍历效率高

## 示例

```cpp
#include <spmv/spmv.h>

int main() {
    // 创建 3x3 CSR 矩阵，5 个非零元素
    CSRMatrix* csr = csr_create(3, 3, 5);

    // 从密集数组填充
    float dense[] = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    csr_from_dense(csr, dense, 3, 3);

    // 计算统计信息
    CSRStats stats = csr_compute_stats(csr);
    printf("Avg NNZ/row: %.2f\n", stats.avg_nnz_per_row);
    printf("Skewness: %.2f\n", stats.skewness);

    // 传输到 GPU
    csr_to_gpu(csr);

    // ... 用于 SpMV 计算 ...

    csr_destroy(csr);
    return 0;
}
```
