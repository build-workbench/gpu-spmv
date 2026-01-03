#ifndef SPMV_ELL_MATRIX_H
#define SPMV_ELL_MATRIX_H

#include "common.h"
#include "csr_matrix.h"
#include <cstddef>

namespace spmv {

// ELL (ELLPACK) 格式稀疏矩阵
// Column-major 存储以实现 GPU 合并访问
struct ELLMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int max_nnz_per_row;    // 每行最大非零元素数
    
    // Column-major 存储: values[k * num_rows + row]
    float* values;          // 值数组 [num_rows * max_nnz_per_row]
    int* col_indices;       // 列索引 [num_rows * max_nnz_per_row], -1 表示填充
    
    // GPU 端指针
    float* d_values;
    int* d_col_indices;
    
    // 标记是否拥有内存
    bool owns_host_memory;
    bool owns_device_memory;
};

// 创建 ELL 矩阵
ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row);

// 销毁 ELL 矩阵
void ell_destroy(ELLMatrix* mat);

// 从稠密矩阵转换为 ELL 格式
int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols);

// 从 CSR 格式转换为 ELL 格式
int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr);

// 将 ELL 矩阵转换回稠密矩阵
int ell_to_dense(const ELLMatrix* ell, float* dense);

// 查询元素值
float ell_get_element(const ELLMatrix* mat, int row, int col);

// 传输到 GPU
int ell_to_gpu(ELLMatrix* mat);

// 从 GPU 传输回主机
int ell_from_gpu(ELLMatrix* mat);

// 释放 GPU 内存
void ell_free_gpu(ELLMatrix* mat);

// 序列化到文件
int ell_serialize(const ELLMatrix* mat, const char* filename);

// 从文件反序列化
int ell_deserialize(ELLMatrix* mat, const char* filename);

// 获取 Column-major 索引
inline int ell_index(int row, int k, int num_rows) {
    return k * num_rows + row;
}

} // namespace spmv

#endif // SPMV_ELL_MATRIX_H
