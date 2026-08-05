#include "spmv/market_io.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "internal/csr_device.h"

namespace spmv {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

int csr_read_matrix_market(CSRMatrix* mat, const char* filename) {
    if (!mat || !filename) {
        return static_cast<int>(SpMVError::INVALID_ARGUMENT);
    }

    std::ifstream file(filename);
    if (!file) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    std::string header;
    if (!std::getline(file, header)) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    std::istringstream header_stream(to_lower(header));
    std::string banner, object, format, field, symmetry;
    header_stream >> banner >> object >> format >> field >> symmetry;
    if (banner != "%%matrixmarket" || object != "matrix" || format != "coordinate") {
        return static_cast<int>(SpMVError::INVALID_FORMAT);
    }
    if (field != "real" && field != "integer" && field != "pattern") {
        return static_cast<int>(SpMVError::INVALID_FORMAT);  // complex unsupported
    }
    if (symmetry != "general" && symmetry != "symmetric") {
        return static_cast<int>(SpMVError::INVALID_FORMAT);  // hermitian/skew unsupported
    }
    const bool symmetric = (symmetry == "symmetric");
    const bool pattern = (field == "pattern");
    const bool integer = (field == "integer");

    // Skip comment lines.
    std::string line;
    while (file.peek() == '%') {
        if (!std::getline(file, line)) {
            return static_cast<int>(SpMVError::FILE_IO);
        }
    }

    long long num_rows = 0, num_cols = 0, declared_nnz = 0;
    if (!std::getline(file, line)) {
        return static_cast<int>(SpMVError::FILE_IO);
    }
    {
        std::istringstream size_stream(line);
        if (!(size_stream >> num_rows >> num_cols >> declared_nnz)) {
            return static_cast<int>(SpMVError::FILE_IO);
        }
    }
    if (num_rows < 0 || num_cols < 0 || declared_nnz < 0 || num_rows > INT_MAX ||
        num_cols > INT_MAX) {
        return static_cast<int>(SpMVError::FILE_IO);
    }

    using Entry = std::tuple<int, int, float>;
    std::vector<Entry> entries;
    entries.reserve(static_cast<size_t>(declared_nnz) * (symmetric ? 2 : 1));
    for (long long i = 0; i < declared_nnz; i++) {
        if (!std::getline(file, line)) {
            return static_cast<int>(SpMVError::FILE_IO);
        }
        std::istringstream entry_stream(line);
        long long r = 0, c = 0;
        float value = 1.0f;
        if (!(entry_stream >> r >> c)) {
            return static_cast<int>(SpMVError::FILE_IO);
        }
        if (!pattern) {
            if (integer) {
                long long iv = 0;
                if (!(entry_stream >> iv)) {
                    return static_cast<int>(SpMVError::FILE_IO);
                }
                value = static_cast<float>(iv);
            } else if (!(entry_stream >> value)) {
                return static_cast<int>(SpMVError::FILE_IO);
            }
        }
        if (r < 1 || c < 1 || r > num_rows || c > num_cols) {
            return static_cast<int>(SpMVError::FILE_IO);
        }
        entries.emplace_back(static_cast<int>(r - 1), static_cast<int>(c - 1), value);
        if (symmetric && r != c) {
            entries.emplace_back(static_cast<int>(c - 1), static_cast<int>(r - 1), value);
        }
    }

    // Sort by (row, col) and sum duplicate entries.
    std::sort(entries.begin(), entries.end());
    std::vector<Entry> merged;
    merged.reserve(entries.size());
    for (const Entry& entry : entries) {
        if (!merged.empty() && std::get<0>(merged.back()) == std::get<0>(entry) &&
            std::get<1>(merged.back()) == std::get<1>(entry)) {
            std::get<2>(merged.back()) += std::get<2>(entry);
        } else {
            merged.push_back(entry);
        }
    }

    const int rows = static_cast<int>(num_rows);
    const int cols = static_cast<int>(num_cols);
    const int nnz = static_cast<int>(merged.size());

    // Host-side mutation invalidates the device mirror.
    csr_free_device_data(mat);

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

    std::fill(mat->row_ptrs, mat->row_ptrs + rows + 1, 0);
    for (int i = 0; i < nnz; i++) {
        mat->row_ptrs[std::get<0>(merged[i]) + 1]++;
        mat->col_indices[i] = std::get<1>(merged[i]);
        mat->values[i] = std::get<2>(merged[i]);
    }
    for (int i = 0; i < rows; i++) {
        mat->row_ptrs[i + 1] += mat->row_ptrs[i];
    }

    return static_cast<int>(SpMVError::SUCCESS);
}

}  // namespace spmv
