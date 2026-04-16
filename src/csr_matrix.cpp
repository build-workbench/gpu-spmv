#include "spmv/csr_matrix.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace spmv {

CSRMatrix* csr_create(int rows, int cols, int nnz) {
    if (rows < 0 || cols < 0 || nnz < 0) {
        return nullptr;
    }

    CSRMatrix* mat = new CSRMatrix();
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->nnz = nnz;

    mat->values = (nnz > 0) ? new float[nnz]() : nullptr;
    mat->col_indices = (nnz > 0) ? new int[nnz]() : nullptr;
    mat->row_ptrs = new int[rows + 1]();

    mat->d_values = nullptr;
    mat->d_col_indices = nullptr;
    mat->d_row_ptrs = nullptr;

    mat->owns_host_memory = true;
    mat->owns_device_memory = false;

    return mat;
}

void csr_destroy(CSRMatrix* mat) {
    if (!mat)
        return;

    if (mat->owns_host_memory) {
        delete[] mat->values;
        delete[] mat->col_indices;
        delete[] mat->row_ptrs;
    }

    if (mat->owns_device_memory) {
        csr_free_gpu(mat);
    }

    delete mat;
}

int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols) {
    if (!csr || !dense || rows <= 0 || cols <= 0) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // Check for potential overflow in size calculation
    if (rows > INT_MAX / cols) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // 主机端内容变化后，旧的 GPU 镜像立即失效
    csr_free_gpu(csr);

    // 首先计算非零元素数量
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (dense[i] != 0.0f) {
            nnz++;
        }
    }

    // 释放旧内存
    if (csr->owns_host_memory) {
        delete[] csr->values;
        delete[] csr->col_indices;
        delete[] csr->row_ptrs;
    }

    // 分配新内存
    csr->num_rows = rows;
    csr->num_cols = cols;
    csr->nnz = nnz;
    csr->values = (nnz > 0) ? new float[nnz] : nullptr;
    csr->col_indices = (nnz > 0) ? new int[nnz] : nullptr;
    csr->row_ptrs = new int[rows + 1];
    csr->owns_host_memory = true;

    // 填充 CSR 数据
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        csr->row_ptrs[i] = idx;
        for (int j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (val != 0.0f) {
                csr->values[idx] = val;
                csr->col_indices[idx] = j;
                idx++;
            }
        }
    }
    csr->row_ptrs[rows] = nnz;

    return static_cast<int>(SpMVError::SUCCESS);
}

int csr_to_dense(const CSRMatrix* csr, float* dense) {
    if (!csr || !dense) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // Check for potential overflow in size calculation
    size_t total_size = static_cast<size_t>(csr->num_rows) * static_cast<size_t>(csr->num_cols);
    if (total_size > static_cast<size_t>(INT_MAX)) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // 初始化为零
    std::memset(dense, 0, csr->num_rows * csr->num_cols * sizeof(float));

    // 填充非零元素
    for (int i = 0; i < csr->num_rows; i++) {
        for (int j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
            int col = csr->col_indices[j];
            dense[i * csr->num_cols + col] = csr->values[j];
        }
    }

    return static_cast<int>(SpMVError::SUCCESS);
}

float csr_get_element(const CSRMatrix* mat, int row, int col) {
    if (!mat || row < 0 || row >= mat->num_rows || col < 0 || col >= mat->num_cols) {
        return 0.0f;
    }

    // 在该行中线性扫描列索引（列索引有序，可提前终止）
    int start = mat->row_ptrs[row];
    int end = mat->row_ptrs[row + 1];

    for (int i = start; i < end; i++) {
        if (mat->col_indices[i] == col) {
            return mat->values[i];
        }
        if (mat->col_indices[i] > col) {
            break;  // 列索引是有序的
        }
    }

    return 0.0f;
}

int csr_to_gpu(CSRMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (!mat->row_ptrs || (mat->nnz > 0 && (!mat->values || !mat->col_indices))) {
        return static_cast<int>(SpMVError::INVALID_FORMAT);
    }

    float* new_d_values = nullptr;
    int* new_d_col_indices = nullptr;
    int* new_d_row_ptrs = nullptr;

    auto cleanup_partial_allocations = [&]() {
        if (new_d_values) {
            cudaFree(new_d_values);
            new_d_values = nullptr;
        }
        if (new_d_col_indices) {
            cudaFree(new_d_col_indices);
            new_d_col_indices = nullptr;
        }
        if (new_d_row_ptrs) {
            cudaFree(new_d_row_ptrs);
            new_d_row_ptrs = nullptr;
        }
    };

    if (mat->nnz > 0) {
        cudaError_t err =
            cudaMalloc(reinterpret_cast<void**>(&new_d_values), mat->nnz * sizeof(float));
        if (err != cudaSuccess) {
            cleanup_partial_allocations();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        err = cudaMalloc(reinterpret_cast<void**>(&new_d_col_indices), mat->nnz * sizeof(int));
        if (err != cudaSuccess) {
            cleanup_partial_allocations();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }
    }

    cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&new_d_row_ptrs), (mat->num_rows + 1) * sizeof(int));
    if (err != cudaSuccess) {
        cleanup_partial_allocations();
        return static_cast<int>(SpMVError::CUDA_MALLOC);
    }

    if (mat->nnz > 0) {
        err =
            cudaMemcpy(new_d_values, mat->values, mat->nnz * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup_partial_allocations();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }

        err = cudaMemcpy(new_d_col_indices, mat->col_indices, mat->nnz * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup_partial_allocations();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }
    }

    err = cudaMemcpy(new_d_row_ptrs, mat->row_ptrs, (mat->num_rows + 1) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup_partial_allocations();
        return static_cast<int>(SpMVError::CUDA_MEMCPY);
    }

    csr_free_gpu(mat);
    mat->d_values = new_d_values;
    mat->d_col_indices = new_d_col_indices;
    mat->d_row_ptrs = new_d_row_ptrs;
    mat->owns_device_memory = true;

    return static_cast<int>(SpMVError::SUCCESS);
}

int csr_from_gpu(CSRMatrix* mat) {
    if (!mat || !mat->d_row_ptrs) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (mat->nnz > 0 && mat->d_values && mat->d_col_indices) {
        CUDA_CHECK_MEMCPY(cudaMemcpy(mat->values, mat->d_values, mat->nnz * sizeof(float),
                                     cudaMemcpyDeviceToHost));
        CUDA_CHECK_MEMCPY(cudaMemcpy(mat->col_indices, mat->d_col_indices, mat->nnz * sizeof(int),
                                     cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_MEMCPY(cudaMemcpy(mat->row_ptrs, mat->d_row_ptrs, (mat->num_rows + 1) * sizeof(int),
                                 cudaMemcpyDeviceToHost));

    return static_cast<int>(SpMVError::SUCCESS);
}

void csr_free_gpu(CSRMatrix* mat) {
    if (!mat)
        return;

    if (mat->d_values) {
        cudaFree(mat->d_values);
        mat->d_values = nullptr;
    }
    if (mat->d_col_indices) {
        cudaFree(mat->d_col_indices);
        mat->d_col_indices = nullptr;
    }
    if (mat->d_row_ptrs) {
        cudaFree(mat->d_row_ptrs);
        mat->d_row_ptrs = nullptr;
    }
    mat->owns_device_memory = false;
}

int csr_serialize(const CSRMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    // 写入头部信息
    file.write(reinterpret_cast<const char*>(&mat->num_rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->num_cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->nnz), sizeof(int));

    // 写入数据
    if (mat->nnz > 0) {
        file.write(reinterpret_cast<const char*>(mat->values), mat->nnz * sizeof(float));
        file.write(reinterpret_cast<const char*>(mat->col_indices), mat->nnz * sizeof(int));
    }
    file.write(reinterpret_cast<const char*>(mat->row_ptrs), (mat->num_rows + 1) * sizeof(int));

    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    return static_cast<int>(SpMVError::SUCCESS);
}

int csr_deserialize(CSRMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    // 读取头部信息
    int rows, cols, nnz;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(int));

    if (!file || rows < 0 || cols < 0 || nnz < 0) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    // 主机端内容变化后，旧的 GPU 镜像立即失效
    csr_free_gpu(mat);

    // 释放旧内存
    if (mat->owns_host_memory) {
        delete[] mat->values;
        delete[] mat->col_indices;
        delete[] mat->row_ptrs;
    }

    // 分配新内存
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->nnz = nnz;
    mat->values = (nnz > 0) ? new float[nnz] : nullptr;
    mat->col_indices = (nnz > 0) ? new int[nnz] : nullptr;
    mat->row_ptrs = new int[rows + 1];
    mat->owns_host_memory = true;

    // 读取数据
    if (nnz > 0) {
        file.read(reinterpret_cast<char*>(mat->values), nnz * sizeof(float));
        file.read(reinterpret_cast<char*>(mat->col_indices), nnz * sizeof(int));
    }
    file.read(reinterpret_cast<char*>(mat->row_ptrs), (rows + 1) * sizeof(int));

    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    return static_cast<int>(SpMVError::SUCCESS);
}

CSRStats csr_compute_stats(const CSRMatrix* mat) {
    CSRStats stats = {0.0f, 0, INT_MAX, 0.0f};

    if (!mat || mat->num_rows == 0) {
        stats.min_nnz_per_row = 0;
        return stats;
    }

    stats.avg_nnz_per_row = static_cast<float>(mat->nnz) / mat->num_rows;

    for (int i = 0; i < mat->num_rows; i++) {
        int row_nnz = mat->row_ptrs[i + 1] - mat->row_ptrs[i];
        stats.max_nnz_per_row = std::max(stats.max_nnz_per_row, row_nnz);
        stats.min_nnz_per_row = std::min(stats.min_nnz_per_row, row_nnz);
    }

    stats.skewness = static_cast<float>(stats.max_nnz_per_row) / (stats.min_nnz_per_row + 1);

    return stats;
}

}  // namespace spmv
