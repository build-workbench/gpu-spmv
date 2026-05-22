#include "ell_device.h"

namespace spmv {

namespace {

struct ELLMatrixInternal {
    float* d_values = nullptr;
    int* d_col_indices = nullptr;
};

ELLMatrixInternal* get_internal(ELLMatrix* mat) {
    return mat ? static_cast<ELLMatrixInternal*>(mat->internal) : nullptr;
}

const ELLMatrixInternal* get_internal(const ELLMatrix* mat) {
    return mat ? static_cast<const ELLMatrixInternal*>(mat->internal) : nullptr;
}

void free_device(ELLMatrixInternal* internal) {
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
}

}  // namespace

void* ell_create_device_state() {
    return new ELLMatrixInternal();
}

void ell_destroy_device_state(ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    if (!internal) {
        return;
    }
    free_device(internal);
    delete internal;
    mat->internal = nullptr;
}

float* ell_d_values(ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_values : nullptr;
}

const float* ell_d_values(const ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_values : nullptr;
}

int* ell_d_col_indices(ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_col_indices : nullptr;
}

const int* ell_d_col_indices(const ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal ? internal->d_col_indices : nullptr;
}

bool ell_has_device_data(const ELLMatrix* mat) {
    auto* internal = get_internal(mat);
    return internal && internal->d_values != nullptr;
}

void ell_free_device_data(ELLMatrix* mat) {
    free_device(get_internal(mat));
}

int ell_upload_device_data(ELLMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0 && (!mat->values || !mat->col_indices)) {
        return static_cast<int>(SpMVError::INVALID_FORMAT);
    }

    auto* internal = get_internal(mat);
    if (!internal) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    free_device(internal);

    float* new_d_values = nullptr;
    int* new_d_col_indices = nullptr;

    auto cleanup = [&]() {
        if (new_d_values) {
            cudaFree(new_d_values);
        }
        if (new_d_col_indices) {
            cudaFree(new_d_col_indices);
        }
    };

    if (size > 0) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_d_values), size * sizeof(float));
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        err = cudaMalloc(reinterpret_cast<void**>(&new_d_col_indices), size * sizeof(int));
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MALLOC);
        }

        err = cudaMemcpy(new_d_values, mat->values, size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }

        err = cudaMemcpy(new_d_col_indices, mat->col_indices, size * sizeof(int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(SpMVError::CUDA_MEMCPY);
        }
    }

    internal->d_values = new_d_values;
    internal->d_col_indices = new_d_col_indices;
    return static_cast<int>(SpMVError::SUCCESS);
}

int ell_download_device_data(ELLMatrix* mat) {
    if (!mat) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    auto* internal = get_internal(mat);
    size_t size = static_cast<size_t>(mat->num_rows) * mat->max_nnz_per_row;
    if (size > 0 && internal && internal->d_values && internal->d_col_indices) {
        if (!mat->values || !mat->col_indices) {
            return static_cast<int>(SpMVError::INVALID_ARGUMENT);
        }
        CUDA_CHECK_MEMCPY(
            cudaMemcpy(mat->values, internal->d_values, size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_MEMCPY(cudaMemcpy(mat->col_indices, internal->d_col_indices, size * sizeof(int),
                                     cudaMemcpyDeviceToHost));
    }

    return static_cast<int>(SpMVError::SUCCESS);
}

}  // namespace spmv
