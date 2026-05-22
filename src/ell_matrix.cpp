#include "internal/ell_device.h"
#include "spmv/ell_matrix.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <fstream>
#include <new>

namespace spmv {

// Public API -----------------------------------------------------------------

ELLMatrix* ell_create(int rows, int cols, int max_nnz_per_row) {
    if (rows < 0 || cols < 0 || max_nnz_per_row < 0) {
        return nullptr;
    }

    ELLMatrix* mat = new ELLMatrix();
    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->max_nnz_per_row = max_nnz_per_row;
    mat->nnz = 0;

    size_t size = static_cast<size_t>(rows) * max_nnz_per_row;
    mat->values = (size > 0) ? new float[size]() : nullptr;
    mat->col_indices = (size > 0) ? new int[size]() : nullptr;

    if (mat->col_indices) {
        for (size_t i = 0; i < size; i++) {
            mat->col_indices[i] = -1;
        }
    }

    mat->internal = ell_create_device_state();

    return mat;
}

void ell_destroy(ELLMatrix* mat) {
    if (!mat)
        return;

    delete[] mat->values;
    delete[] mat->col_indices;

    ell_destroy_device_state(mat);

    delete mat;
}

int ell_from_dense(ELLMatrix* ell, const float* dense, int rows, int cols) {
    if (!ell || !dense || rows <= 0 || cols <= 0) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (rows > INT_MAX / cols) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    ell_free_device_data(ell);

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

    delete[] ell->values;
    delete[] ell->col_indices;

    ell->num_rows = rows;
    ell->num_cols = cols;
    ell->max_nnz_per_row = max_nnz;

    size_t size = static_cast<size_t>(rows) * max_nnz;
    ell->values = (size > 0) ? new float[size]() : nullptr;
    ell->col_indices = (size > 0) ? new int[size]() : nullptr;

    if (ell->col_indices) {
        for (size_t i = 0; i < size; i++) {
            ell->col_indices[i] = -1;
            ell->values[i] = 0.0f;
        }
    }

    int total_nnz = 0;
    for (int i = 0; i < rows; i++) {
        int k = 0;
        for (int j = 0; j < cols; j++) {
            float val = dense[i * cols + j];
            if (val != 0.0f) {
                int idx = ell_index(i, k, rows);
                ell->values[idx] = val;
                ell->col_indices[idx] = j;
                k++;
                total_nnz++;
            }
        }
    }
    ell->nnz = total_nnz;

    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_from_csr(ELLMatrix* ell, const CSRMatrix* csr) {
    if (!ell || !csr) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    if (!csr->row_ptrs || (csr->nnz > 0 && (!csr->values || !csr->col_indices))) {
        return static_cast<int>(SpMVError::INVALID_FORMAT);
    }

    ell_free_device_data(ell);

    int max_nnz = 0;
    for (int i = 0; i < csr->num_rows; i++) {
        int row_nnz = csr->row_ptrs[i + 1] - csr->row_ptrs[i];
        max_nnz = std::max(max_nnz, row_nnz);
    }

    delete[] ell->values;
    delete[] ell->col_indices;

    ell->num_rows = csr->num_rows;
    ell->num_cols = csr->num_cols;
    ell->max_nnz_per_row = max_nnz;

    size_t size = static_cast<size_t>(csr->num_rows) * max_nnz;
    ell->values = (size > 0) ? new float[size]() : nullptr;
    ell->col_indices = (size > 0) ? new int[size]() : nullptr;

    if (ell->col_indices) {
        for (size_t i = 0; i < size; i++) {
            ell->col_indices[i] = -1;
            ell->values[i] = 0.0f;
        }
    }

    for (int i = 0; i < csr->num_rows; i++) {
        int k = 0;
        for (int j = csr->row_ptrs[i]; j < csr->row_ptrs[i + 1]; j++) {
            int idx = ell_index(i, k, csr->num_rows);
            ell->values[idx] = csr->values[j];
            ell->col_indices[idx] = csr->col_indices[j];
            k++;
        }
    }
    ell->nnz = csr->nnz;

    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_to_dense(const ELLMatrix* ell, float* dense) {
    if (!ell || !dense) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    size_t total_size = static_cast<size_t>(ell->num_rows) * static_cast<size_t>(ell->num_cols);
    if (total_size > static_cast<size_t>(INT_MAX)) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::memset(dense, 0, ell->num_rows * ell->num_cols * sizeof(float));

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
            break;
        }
    }

    return 0.0f;
}

int ell_to_gpu(ELLMatrix* mat) { return ell_upload_device_data(mat); }

int ell_from_gpu(ELLMatrix* mat) { return ell_download_device_data(mat); }

int ell_serialize(const ELLMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    const uint32_t magic = 0x4C4C4553;  // "SELL" in little-endian
    const uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

    file.write(reinterpret_cast<const char*>(&mat->num_rows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->num_cols), sizeof(int));
    file.write(reinterpret_cast<const char*>(&mat->max_nnz_per_row), sizeof(int));

    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0) {
        file.write(reinterpret_cast<const char*>(mat->values), size * sizeof(float));
        file.write(reinterpret_cast<const char*>(mat->col_indices), size * sizeof(int));
    }

    uint64_t checksum = 0;
    checksum += static_cast<uint64_t>(mat->num_rows);
    checksum += static_cast<uint64_t>(mat->num_cols);
    checksum += static_cast<uint64_t>(mat->max_nnz_per_row);
    for (size_t i = 0; i < size; i++) {
        checksum += static_cast<uint64_t>(mat->col_indices[i]);
    }
    file.write(reinterpret_cast<const char*>(&checksum), sizeof(uint64_t));

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

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    if (!file || magic != 0x4C4C4553) {  // "SELL"
        return static_cast<int>(SpMVError::FILE_IO);
    }

    if (version > 1) {
        fprintf(stderr, "Warning: ELL file version %u is newer than supported version 1\n",
                version);
    }

    int rows, cols, max_nnz;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));
    file.read(reinterpret_cast<char*>(&max_nnz), sizeof(int));

    if (!file || rows < 0 || cols < 0 || max_nnz < 0) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    ell_free_device_data(mat);

    delete[] mat->values;
    delete[] mat->col_indices;

    mat->num_rows = rows;
    mat->num_cols = cols;
    mat->max_nnz_per_row = max_nnz;

    size_t size = static_cast<size_t>(rows) * max_nnz;
    mat->values = (size > 0) ? new (std::nothrow) float[size] : nullptr;
    mat->col_indices = (size > 0) ? new (std::nothrow) int[size] : nullptr;

    if (size > 0 && (!mat->values || !mat->col_indices)) {
        delete[] mat->values;
        delete[] mat->col_indices;
        mat->values = nullptr;
        mat->col_indices = nullptr;
        return static_cast<int>(SpMVError::OUT_OF_MEMORY);
    }

    if (size > 0) {
        file.read(reinterpret_cast<char*>(mat->values), size * sizeof(float));
        file.read(reinterpret_cast<char*>(mat->col_indices), size * sizeof(int));
    }

    uint64_t stored_checksum;
    file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(uint64_t));

    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    uint64_t computed_checksum = 0;
    computed_checksum += static_cast<uint64_t>(rows);
    computed_checksum += static_cast<uint64_t>(cols);
    computed_checksum += static_cast<uint64_t>(max_nnz);
    for (size_t i = 0; i < size; i++) {
        computed_checksum += static_cast<uint64_t>(mat->col_indices[i]);
    }

    if (computed_checksum != stored_checksum) {
        fprintf(stderr, "ELL file checksum mismatch: expected %lu, got %lu\n", stored_checksum,
                computed_checksum);
        return static_cast<int>(SpMVError::FILE_IO);
    }

    int total_nnz = 0;
    for (size_t i = 0; i < size; i++) {
        if (mat->col_indices[i] >= 0)
            total_nnz++;
    }
    mat->nnz = total_nnz;

    return static_cast<int>(SpMVError::SUCCESS);
}

bool ell_validate(const ELLMatrix* mat) {
    if (!mat) {
        return false;
    }

    if (mat->num_rows < 0 || mat->num_cols < 0 || mat->max_nnz_per_row < 0 || mat->nnz < 0) {
        return false;
    }

    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;

    if (size > 0 && (!mat->values || !mat->col_indices)) {
        return false;
    }

    int actual_nnz = 0;
    for (size_t i = 0; i < size; i++) {
        int col = mat->col_indices[i];
        if (col == -1) {
            // Padding entry
        } else if (col < 0 || col >= mat->num_cols) {
            return false;
        } else {
            actual_nnz++;
        }
    }

    if (actual_nnz != mat->nnz) {
        return false;
    }

    return true;
}

}  // namespace spmv
