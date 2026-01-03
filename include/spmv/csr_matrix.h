#ifndef SPMV_CSR_MATRIX_H
#define SPMV_CSR_MATRIX_H

#include "common.h"
#include <cstddef>
#include <vector>

namespace spmv {

// CSR (Compressed Sparse Row) 格式稀疏矩阵
struct CSRMatrix {
    int num_rows;           // 矩阵行数
    int num_cols;           // 矩阵列数
    int nnz;                // 非零元素总数
    
    float* values;          // 非零元素值数组 [nnz]
    int* col_indices;       // 列索引数组 [nnz]
    int* row_ptrs;          // 行指针数组 [num_rows + 1]
    
    // GPU 端指针
    float* d_values;
    int* d_col_indices;
    int* d_row_ptrs;
    
    // 标记是否拥有内存
    bool owns_host_memory;
    bool owns_device_memory;
};

// 创建 CSR 矩阵
CSRMatrix* csr_create(int rows, int cols, int nnz);

// 销毁 CSR 矩阵
void csr_destroy(CSRMatrix* mat);

// 从稠密矩阵转换为 CSR 格式
// dense: 行优先存储的稠密矩阵 [rows * cols]
// 返回: 0 成功, 负数错误码
int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols);

// 将 CSR 矩阵转换回稠密矩阵
// dense: 输出缓冲区 [num_rows * num_cols]
int csr_to_dense(const CSRMatrix* csr, float* dense);

// 查询元素值
float csr_get_element(const CSRMatrix* mat, int row, int col);

// 传输到 GPU
int csr_to_gpu(CSRMatrix* mat);

// 从 GPU 传输回主机
int csr_from_gpu(CSRMatrix* mat);

// 释放 GPU 内存
void csr_free_gpu(CSRMatrix* mat);

// 序列化到文件
int csr_serialize(const CSRMatrix* mat, const char* filename);

// 从文件反序列化
int csr_deserialize(CSRMatrix* mat, const char* filename);

// 计算每行非零元素数量的统计信息
struct CSRStats {
    float avg_nnz_per_row;
    int max_nnz_per_row;
    int min_nnz_per_row;
    float skewness;  // max / (min + 1)
};

CSRStats csr_compute_stats(const CSRMatrix* mat);

} // namespace spmv

#endif // SPMV_CSR_MATRIX_H
