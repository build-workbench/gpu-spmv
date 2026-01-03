#include "spmv/ell_matrix.h"
#include <cstring>
#include <fstream>
#include <algorithm>

namespace spmv {

ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row) {
    if (rows < 0 || cols < 0 || max_nnz_per_row < 0) {
        return nullptr;
    }
    
    ELLMatrix* mat = new ELLMatrix();
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->max_nnz_per_row = max_nnz_per_row;
    
    size_t size = static_cast<size_t>(rows) * max_nnz_per_row;
    mat->values = (size > 0) ? new float[size]() : nullptr;
    mat->col_indices = (size > 0) ? new int[size]() : nullptr;
    
    // 初始化列索引为 -1 (表示填充)
    if (mat->col_indices) {
        for (size_t i = 0; i < size; i++) {
            mat->col_indices[i] = -1;
        }
    }
    
    mat->d_values = nullptr;
    mat->d_col_indices = nullptr;
    
    mat->owns_host_memory = true;
    mat->owns_device_memory = false;
    
    return mat;
}

void ell_destroy(ELLMatrix* mat) {
    if (!mat) return;
    
    if (mat->owns_host_memory) {
        delete[] mat->values;
        delete[] mat->col_indices;
    }
    
    if (mat->owns_device_memory) {
        ell_free_gpu(mat);
    }
    
    delete mat;
}

int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols) {
    if (!ell || !dense || rows <= 0 || cols <= 0) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    // 计算每行最大非零元素数
    int max_nnz = 0;
    for (int i = 0; i < rows; i++) {
        int row_nnz = 0;
        for (int j = 0; j < cols; j++) {
            if (dense[i * cols + j] != 0.0f) {
                row_nnz++;
            }
        }
        max_nnz = std::max(max_nnz, row_nnz);
    }
    
    // 释放旧内存
    if (ell->owns_host_memory) {
        delete[] ell->values;
        delete[] ell->col_indices;
    }
    
    // 分配新内存
    ell->num_rows = rows;
    ell->num_cols = cols;
    ell->max_nnz_per_row = max_nnz;
    
    size_t size = static_cast<size_t>(rows) * max_nnz;
    ell->values = (size > 0) ? new float[size]() : nullptr;
    ell->col_indices = (size > 0) ? new int[size]() : nullptr;
    ell->owns_host_memory = true;
    
    // 初始化为填充值
    if (ell->col_indices) {
        for (size_t i = 0; i < size; i++) {
            ell->col_indices[i] = -1;
            ell->values[i] = 0.0f;
        }
    }
    
    // 填充 ELL 数据 (Column-major)
    for (int i = 0; i < rows; i++) {
        int k = 0;
        for (int j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (val != 0.0f) {
                int idx = ell_index(i, k, rows);
                ell->values[idx] = val;
                ell->col_indices[idx] = j;
                k++;
            }
        }
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr) {
    if (!ell || !csr) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    // 计算每行最大非零元素数
    int max_nnz = 0;
    for (int i = 0; i < csr->num_rows; i++) {
        int row_nnz = csr->row_ptrs[i + 1] - csr->row_ptrs[i];
        max_nnz = std::max(max_nnz, row_nnz);
    }
    
    // 释放旧内存
    if (ell->owns_host_memory) {
        delete[] ell->values;
        delete[] ell->col_indices;
    }
    
    // 分配新内存
    ell->num_rows = csr->num_rows;
    ell->num_cols = csr->num_cols;
    ell->max_nnz_per_row = max_nnz;
    
    size_t size = static_cast<size_t>(csr->num_rows) * max_nnz;
    ell->values = (size > 0) ? new float[size]() : nullptr;
    ell->col_indices = (size > 0) ? new int[size]() : nullptr;
    ell->owns_host_memory = true;
    
    // 初始化为填充值
    if (ell->col_indices) {
        for (size_t i = 0; i < size; i++) {
            ell->col_indices[i] = -1;
            ell->values[i] = 0.0f;
        }
    }
    
    // 从 CSR 转换 (Column-major)
    for (int i = 0; i < csr->num_rows; i++) {
        int k = 0;
        for (int j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
            int idx = ell_index(i, k, csr->num_rows);
            ell->values[idx] = csr->values[j];
            ell->col_indices[idx] = csr->col_indices[j];
            k++;
        }
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}


int ell_to_dense(const ELLMatrix* ell, float* dense) {
    if (!ell || !dense) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    // 初始化为零
    std::memset(dense, 0, ell->num_rows * ell->num_cols * sizeof(float));
    
    // 填充非零元素
    for (int i = 0; i < ell->num_rows; i++) {
        for (int k = 0; k < ell->max_nnz_per_row; k++) {
            int idx = ell_index(i, k, ell->num_rows);
            int col = ell->col_indices[idx];
            if (col >= 0) {
                dense[i * ell->num_cols + col] = ell->values[idx];
            }
        }
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}

float ell_get_element(const ELLMatrix* mat, int row, int col) {
    if (!mat || row < 0 || row >= mat->num_rows || col < 0 || col >= mat->num_cols) {
        return 0.0f;
    }
    
    for (int k = 0; k < mat->max_nnz_per_row; k++) {
        int idx = ell_index(row, k, mat->num_rows);
        if (mat->col_indices[idx] == col) {
            return mat->values[idx];
        }
        if (mat->col_indices[idx] < 0) {
            break;  // 到达填充区域
        }
    }
    
    return 0.0f;
}

int ell_to_gpu(ELLMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    ell_free_gpu(mat);
    
    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0) {
        CUDA_CHECK(cudaMalloc(&mat->d_values, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mat->d_col_indices, size * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(mat->d_values, mat->values,
                              size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mat->d_col_indices, mat->col_indices,
                              size * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    mat->owns_device_memory = true;
    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_from_gpu(ELLMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0 && mat->d_values && mat->d_col_indices) {
        CUDA_CHECK(cudaMemcpy(mat->values, mat->d_values,
                              size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mat->col_indices, mat->d_col_indices,
                              size * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}

void ell_free_gpu(ELLMatrix* mat) {
    if (!mat) return;
    
    if (mat->d_values) {
        cudaFree(mat->d_values);
        mat->d_values = nullptr;
    }
    if (mat->d_col_indices) {
        cudaFree(mat->d_col_indices);
        mat->d_col_indices = nullptr;
    }
    mat->owns_device_memory = false;
}

int ell_serialize(const ELLMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    
    file.write(reinterpret_cast<const char*>(&mat->num_rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->num_cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->max_nnz_per_row), sizeof(int));
    
    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0) {
        file.write(reinterpret_cast<const char*>(mat->values), size * sizeof(float));
        file.write(reinterpret_cast<const char*>(mat->col_indices), size * sizeof(int));
    }
    
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_deserialize(ELLMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    
    int rows, cols, max_nnz;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&max_nnz), sizeof(int));
    
    if (!file || rows < 0 || cols < 0 || max_nnz < 0) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    
    if (mat->owns_host_memory) {
        delete[] mat->values;
        delete[] mat->col_indices;
    }
    
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->max_nnz_per_row = max_nnz;
    
    size_t size = static_cast<size_t>(rows) * max_nnz;
    mat->values = (size > 0) ? new float[size] : nullptr;
    mat->col_indices = (size > 0) ? new int[size] : nullptr;
    mat->owns_host_memory = true;
    
    if (size > 0) {
        file.read(reinterpret_cast<char*>(mat->values), size * sizeof(float));
        file.read(reinterpret_cast<char*>(mat->col_indices), size * sizeof(int));
    }
    
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    
    return static_cast<int>(SpMVError::SUCCESS);
}

} // namespace spmv
