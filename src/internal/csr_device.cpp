#include "csr_device.h"

namespace spmv {

namespace {

struct CSRMatrixInternal {
    float* d_values = nullptr;
    int* d_col_indices = nullptr;
    int* d_row_ptrs = nullptr;
};

CSRMatrixInternal* get_internal(CSRMatrix* mat) {
    return mat ? static_cast<CSRMatrixInternal*>(mat->internal) : nullptr;
}

const CSRMatrixInternal* get_internal(const CSRMatrix* mat) {
    return mat ? static_cast<const CSRMatrixInternal*>(mat->internal) : nullptr;
}

void free_device(CSRMatrixInternal* internal) {
    if (!internal) {
        return;
    }
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

}  // namespace

void* csr_create_device_state() {
    return new CSRMatrixInternal();
}

void csr_destroy_device_state(CSRMatrix* mat) {
    auto* internal = get_internal(mat);
    if (!internal) {
        return;
    }
    free_device(internal);
    delete internal;
    mat->internal = nullptr;
}

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

int csr_upload_device_data(CSRMatrix* mat) {
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

    free_device(internal);

    float* new_d_values = nullptr;
    int* new_d_col_indices = nullptr;
    int* new_d_row_ptrs = nullptr;

    auto cleanup = [&]() {
        if (new_d_values) {
            cudaFree(new_d_values);
        }
        if (new_d_col_indices) {
            cudaFree(new_d_col_indices);
        }
        if (new_d_row_ptrs) {
            cudaFree(new_d_row_ptrs);
        }
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

int csr_download_device_data(CSRMatrix* mat) {
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

}  // namespace spmv
