#include "spmv/csr_matrix.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>

namespace spmv {

// Internal device state -------------------------------------------------------
struct CSRMatrixInternal {
    float* d_values = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_ptrs = nullptr;
};

// Helpers --------------------------------------------------------------------
static CSRMatrixInternal* get_internal(CSRMatrix* mat) {
    return mat ? static_cast<CSRMatrixInternal*>(mat->internal) : nullptr;
}

static const CSRMatrixInternal* get_internal(const CSRMatrix* mat) {
    return mat ? static_cast<const CSRMatrixInternal*>(mat->internal) : nullptr;
}

static void free_device(CSRMatrixInternal* internal) {
    if (!internal)
        return;
    if (internal->d_values) {
        cudaFree(internal->d_values);
        internal->d_values = nullptr;
    }
    if (internal->d_col_indices) {
        cudaFree(internal->d_col_indices);
        internal->d_col_indices = nullptr;
    }
    if (internal->d_row_ptrs) {
        cudaFree(internal->d_row_ptrs);
        internal->d_row_ptrs = nullptr;
    }
}

// Internal API (used by src/ only) -------------------------------------------
float* csr_d_values(CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_values : nullptr;
}

const float* csr_d_values(const CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_values : nullptr;
}

int* csr_d_col_indices(CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_col_indices : nullptr;
}

const int* csr_d_col_indices(const CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_col_indices : nullptr;
}

int* csr_d_row_ptrs(CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_row_ptrs : nullptr;
}

const int* csr_d_row_ptrs(const CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_row_ptrs : nullptr;
}

bool csr_has_device_data(const CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal && internal->d_row_ptrs != nullptr;
}

void csr_free_device_data(CSRMatrix* mat) {
    free_device(get_internal(mat));
}

// Public API -----------------------------------------------------------------

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
    mat->internal = new CSRMatrixInternal();

    return mat;
}

void csr_destroy(CSRMatrix* mat) {
    if (!mat)
        return;

    delete[] mat->values;
    delete[] mat->col_indices;
    delete[] mat->row_ptrs;

    auto* internal = get_internal(mat);
    if (internal) {
        free_device(internal);
        delete internal;
    }

    delete mat;
}

int csr_from_dense(CSRMatrix* csr, const float* dense, int rows, int cols) {
    if (!csr || !dense || rows <= 0 || cols <= 0) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (rows > INT_MAX / cols) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // Host-side mutation invalidates the device mirror immediately.
    free_device(get_internal(csr));

    int nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (dense[i] != 0.0f) {
            nnz++;
        }
    }

    delete[] csr->values;
    delete[] csr->col_indices;
    delete[] csr->row_ptrs;

    csr->num_rows = rows;
    csr->num_cols = cols;
    csr->nnz = nnz;
    csr->values = (nnz > 0) ? new float[nnz] : nullptr;
    csr->col_indices = (nnz > 0) ? new int[nnz] : nullptr;
    csr->row_ptrs = new int[rows + 1];

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

    size_t total_size = static_cast<size_t>(csr->num_rows) * static_cast<size_t>(csr->num_cols);
    if (total_size > static_cast<size_t>(INT_MAX)) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::memset(dense, 0, csr->num_rows * csr->num_cols * sizeof(float));

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

    int start = mat->row_ptrs[row];
    int end = mat->row_ptrs[row + 1];

    for (int i = start; i < end; i++) {
        if (mat->col_indices[i] == col) {
            return mat->values[i];
        }
        if (mat->col_indices[i] > col) {
            break;
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

    auto* internal = get_internal(mat);
    if (!internal) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    // Free any existing device data before allocating new buffers.
    free_device(internal);

    float* new_d_values = nullptr;
    int* new_d_col_indices = nullptr;
    int* new_d_row_ptrs = nullptr;

    auto cleanup = [&]() {
        if (new_d_values)
            cudaFree(new_d_values);
        if (new_d_col_indices)
            cudaFree(new_d_col_indices);
        if (new_d_row_ptrs)
            cudaFree(new_d_row_ptrs);
    };

    if (mat->nnz > 0) {
        cudaError_t err =
            cudaMalloc(reinterpret_cast<void**>(&new_d_values), mat->nnz * sizeof(float));
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        err = cudaMalloc(reinterpret_cast<void**>(&new_d_col_indices), mat->nnz * sizeof(int));
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }
    }

    cudaError_t err =
        cudaMalloc(reinterpret_cast<void**>(&new_d_row_ptrs), (mat->num_rows + 1) * sizeof(int));
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(SpMVError::CUDA_MALLOC);
    }

    if (mat->nnz > 0) {
        err =
            cudaMemcpy(new_d_values, mat->values, mat->nnz * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }

        err = cudaMemcpy(new_d_col_indices, mat->col_indices, mat->nnz * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }
    }

    err = cudaMemcpy(new_d_row_ptrs, mat->row_ptrs, (mat->num_rows + 1) * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(SpMVError::CUDA_MEMCPY);
    }

    internal->d_values = new_d_values;
    internal->d_col_indices = new_d_col_indices;
    internal->d_row_ptrs = new_d_row_ptrs;

    return static_cast<int>(SpMVError::SUCCESS);
}

int csr_from_gpu(CSRMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    auto* internal = get_internal(mat);
    if (!internal || !internal->d_row_ptrs) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (!mat->row_ptrs) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (mat->nnz > 0 && internal->d_values && internal->d_col_indices) {
        if (!mat->values || !mat->col_indices) {
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        }
        CUDA_CHECK_MEMCPY(cudaMemcpy(mat->values, internal->d_values, mat->nnz * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        CUDA_CHECK_MEMCPY(cudaMemcpy(mat->col_indices, internal->d_col_indices,
                                     mat->nnz * sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK_MEMCPY(cudaMemcpy(mat->row_ptrs, internal->d_row_ptrs,
                                 (mat->num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    return static_cast<int>(SpMVError::SUCCESS);
}

int csr_serialize(const CSRMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    const uint32_t magic = 0x52534353;  // "SCSR" in little-endian
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    file.write(reinterpret_cast<const char*>(&mat->num_rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->num_cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->nnz), sizeof(int));

    if (mat->nnz > 0) {
        file.write(reinterpret_cast<const char*>(mat->values), mat->nnz * sizeof(float));
        file.write(reinterpret_cast<const char*>(mat->col_indices), mat->nnz * sizeof(int));
    }
    file.write(reinterpret_cast<const char*>(mat->row_ptrs), (mat->num_rows + 1) * sizeof(int));

    uint64_t checksum = 0;
    checksum += static_cast<uint64_t>(mat->num_rows);
    checksum += static_cast<uint64_t>(mat->num_cols);
    checksum += static_cast<uint64_t>(mat->nnz);
    for (int i = 0; i < mat->nnz; i++) {
        checksum += static_cast<uint64_t>(mat->col_indices[i]);
    }
    for (int i = 0; i <= mat->num_rows; i++) {
        checksum += static_cast<uint64_t>(mat->row_ptrs[i]);
    }
    file.write(reinterpret_cast<const char*>(&checksum), sizeof(uint64_t));

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

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    if (!file || magic != 0x52534353) {  // "SCSR"
        return static_cast<int>(SpMVError::FILE_IO);
    }

    if (version > 1) {
        fprintf(stderr, "Warning: CSR file version %u is newer than supported version 1\n",
                version);
    }

    int rows, cols, nnz;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(int));

    if (!file || rows < 0 || cols < 0 || nnz < 0) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    // Host-side mutation invalidates device mirror.
    free_device(get_internal(mat));

    delete[] mat->values;
    delete[] mat->col_indices;
    delete[] mat->row_ptrs;

    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->nnz = nnz;
    mat->values = (nnz > 0) ? new (std::nothrow) float[nnz] : nullptr;
    mat->col_indices = (nnz > 0) ? new (std::nothrow) int[nnz] : nullptr;
    mat->row_ptrs = new (std::nothrow) int[rows + 1];

    if ((nnz > 0 && (!mat->values || !mat->col_indices)) || !mat->row_ptrs) {
        delete[] mat->values;
        delete[] mat->col_indices;
        delete[] mat->row_ptrs;
        mat->values = nullptr;
        mat->col_indices = nullptr;
        mat->row_ptrs = nullptr;
        return static_cast<int>(SpMVError::OUT_OF_MEMORY);
    }

    if (nnz > 0) {
        file.read(reinterpret_cast<char*>(mat->values), nnz * sizeof(float));
        file.read(reinterpret_cast<char*>(mat->col_indices), nnz * sizeof(int));
    }
    file.read(reinterpret_cast<char*>(mat->row_ptrs), (rows + 1) * sizeof(int));

    uint64_t stored_checksum;
    file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(uint64_t));

    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    uint64_t computed_checksum = 0;
    computed_checksum += static_cast<uint64_t>(rows);
    computed_checksum += static_cast<uint64_t>(cols);
    computed_checksum += static_cast<uint64_t>(nnz);
    for (int i = 0; i < nnz; i++) {
        computed_checksum += static_cast<uint64_t>(mat->col_indices[i]);
    }
    for (int i = 0; i <= rows; i++) {
        computed_checksum += static_cast<uint64_t>(mat->row_ptrs[i]);
    }

    if (computed_checksum != stored_checksum) {
        fprintf(stderr, "CSR file checksum mismatch: expected %lu, got %lu\n", stored_checksum,
                computed_checksum);
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

bool csr_validate(const CSRMatrix* mat) {
    if (!mat) {
        return false;
    }

    if (mat->num_rows < 0 || mat->num_cols < 0 || mat->nnz < 0) {
        return false;
    }

    if (!mat->row_ptrs) {
        return false;
    }

    if (mat->nnz > 0 && (!mat->values || !mat->col_indices)) {
        return false;
    }

    if (mat->row_ptrs[0] != 0) {
        return false;
    }

    if (mat->row_ptrs[mat->num_rows] != mat->nnz) {
        return false;
    }

    for (int i = 0; i < mat->num_rows; i++) {
        if (mat->row_ptrs[i + 1] < mat->row_ptrs[i]) {
            return false;
        }
    }

    for (int i = 0; i < mat->nnz; i++) {
        if (mat->col_indices[i] < 0 || mat->col_indices[i] >= mat->num_cols) {
            return false;
        }
    }

    return true;
}

}  // namespace spmv
