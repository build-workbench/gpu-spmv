#include "spmv/csr_matrix.h"

#include <climits>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "test_utils.h"

using namespace spmv;
using namespace spmv::test;

class CSRPropertyTest : public ::testing::Test {
   protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 100;
};

// **Feature: spmv-gpu, Property 1: CSR Dense-to-Sparse Round Trip**
// **Validates: Requirements 1.2**
TEST_F(CSRPropertyTest, DenseToSparseRoundTrip) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);

        auto dense_original = generateRandomDenseMatrix(rows, cols, density, rng);

        CSRMatrix* csr = csr_create(0, 0, 0);
        ASSERT_NE(csr, nullptr);

        int result = csr_from_dense(csr, dense_original.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        std::vector<float> dense_reconstructed(rows * cols);
        result = csr_to_dense(csr, dense_reconstructed.data());
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        EXPECT_TRUE(
            floatArraysEqual(dense_original.data(), dense_reconstructed.data(), rows * cols))
            << "Round trip failed at iteration " << iter;

        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 2: CSR Element Lookup Correctness**
// **Validates: Requirements 1.3**
TEST_F(CSRPropertyTest, ElementLookupCorrectness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 50);
        int cols = rng.randInt(1, 50);
        float density = rng.randFloat(0.1f, 0.5f);

        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);

        CSRMatrix* csr = csr_create(0, 0, 0);
        ASSERT_NE(csr, nullptr);

        int result = csr_from_dense(csr, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        // 随机查询多个位置
        for (int q = 0; q < 20; q++) {
            int r = rng.randInt(0, rows - 1);
            int c = rng.randInt(0, cols - 1);

            float expected = dense[r * cols + c];
            float actual = csr_get_element(csr, r, c);

            EXPECT_FLOAT_EQ(expected, actual)
                << "Element lookup failed at (" << r << ", " << c << ") " << "iteration " << iter;
        }

        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 3: CSR Serialization Round Trip**
// **Validates: Requirements 1.5**
TEST_F(CSRPropertyTest, SerializationRoundTrip) {
    std::string test_file_path = getTempFilePath("csr_test.bin");
    const char* test_file = test_file_path.c_str();

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(1, 100);
        int cols = rng.randInt(1, 100);
        float density = rng.randFloat(0.01f, 0.5f);

        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);

        CSRMatrix* csr_original = csr_create(0, 0, 0);
        ASSERT_NE(csr_original, nullptr);

        int result = csr_from_dense(csr_original, dense.data(), rows, cols);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        // 序列化
        result = csr_serialize(csr_original, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        // 反序列化
        CSRMatrix* csr_loaded = csr_create(0, 0, 0);
        result = csr_deserialize(csr_loaded, test_file);
        ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

        // 验证
        EXPECT_EQ(csr_original->num_rows, csr_loaded->num_rows);
        EXPECT_EQ(csr_original->num_cols, csr_loaded->num_cols);
        EXPECT_EQ(csr_original->nnz, csr_loaded->nnz);

        if (csr_original->nnz > 0) {
            EXPECT_TRUE(
                floatArraysEqual(csr_original->values, csr_loaded->values, csr_original->nnz));
            EXPECT_TRUE(intArraysEqual(csr_original->col_indices, csr_loaded->col_indices,
                                       csr_original->nnz));
        }
        EXPECT_TRUE(intArraysEqual(csr_original->row_ptrs, csr_loaded->row_ptrs,
                                   csr_original->num_rows + 1));

        csr_destroy(csr_original);
        csr_destroy(csr_loaded);
    }

    std::remove(test_file);
}

// 单元测试：边界情况
TEST(CSRUnitTest, EmptyMatrix) {
    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    EXPECT_EQ(csr->num_rows, 0);
    EXPECT_EQ(csr->num_cols, 0);
    EXPECT_EQ(csr->nnz, 0);
    csr_destroy(csr);
}

TEST(CSRUnitTest, AllZeroMatrix) {
    std::vector<float> dense(9, 0.0f);  // 3x3 全零矩阵

    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 3, 3);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

    EXPECT_EQ(csr->num_rows, 3);
    EXPECT_EQ(csr->num_cols, 3);
    EXPECT_EQ(csr->nnz, 0);

    csr_destroy(csr);
}

TEST(CSRUnitTest, SingleElementMatrix) {
    std::vector<float> dense = {5.0f};

    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 1, 1);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

    EXPECT_EQ(csr->num_rows, 1);
    EXPECT_EQ(csr->num_cols, 1);
    EXPECT_EQ(csr->nnz, 1);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 5.0f);

    csr_destroy(csr);
}

TEST(CSRUnitTest, GPUTransfer) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};  // 3x3

    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 3, 3);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

    // 传输到 GPU
    result = csr_to_gpu(csr);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));
    EXPECT_TRUE(csr_has_device_data(csr));

    // 修改主机数据
    for (int i = 0; i < csr->nnz; i++) {
        csr->values[i] = 0.0f;
    }

    // 从 GPU 传回
    result = csr_from_gpu(csr);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

    // 验证数据恢复
    std::vector<float> reconstructed(9);
    csr_to_dense(csr, reconstructed.data());

    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(dense[i], reconstructed[i]);
    }

    csr_destroy(csr);
}

TEST(CSRUnitTest, HostMutationInvalidatesDeviceMirror) {
    std::vector<float> dense_a = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    std::vector<float> dense_b = {4, 5, 0, 0, 0, 6, 7, 0, 0};

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense_a.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    ASSERT_EQ(csr_to_gpu(csr), static_cast<int>(SpMVError::SUCCESS));
    EXPECT_TRUE(csr_has_device_data(csr));

    ASSERT_EQ(csr_from_dense(csr, dense_b.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));
    EXPECT_FALSE(csr_has_device_data(csr));

    csr_destroy(csr);
}

// **Feature: spmv-gpu, Matrix Validation**
TEST(CSRUnitTest, ValidateValidMatrix) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};  // 3x3

    CSRMatrix* csr = csr_create(0, 0, 0);
    int result = csr_from_dense(csr, dense.data(), 3, 3);
    ASSERT_EQ(result, static_cast<int>(SpMVError::SUCCESS));

    EXPECT_TRUE(csr_validate(csr));

    csr_destroy(csr);
}

TEST(CSRUnitTest, ValidateNullMatrix) {
    EXPECT_FALSE(csr_validate(nullptr));
}

TEST(CSRUnitTest, ValidateEmptyMatrix) {
    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_NE(csr, nullptr);
    EXPECT_TRUE(csr_validate(csr));
    csr_destroy(csr);
}

// Regression: csr_get_element must not assume sorted column indices.
TEST(CSRUnitTest, GetElementWorksWithUnsortedRows) {
    // Hand-built CSR whose column indices are deliberately unsorted in row 0.
    CSRMatrix* csr = csr_create(2, 3, 3);
    ASSERT_NE(csr, nullptr);
    csr->row_ptrs[0] = 0;
    csr->row_ptrs[1] = 2;
    csr->row_ptrs[2] = 3;
    csr->col_indices[0] = 2;  // row 0: col 2 before col 0
    csr->values[0] = 3.0f;
    csr->col_indices[1] = 0;
    csr->values[1] = 1.0f;
    csr->col_indices[2] = 1;  // row 1
    csr->values[2] = 2.0f;

    EXPECT_TRUE(csr_validate(csr));
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 2), 3.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 1, 1), 2.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 1, 0), 0.0f);

    csr_destroy(csr);
}

// Regression: the version 2 checksum must cover the values array.
TEST(CSRUnitTest, DeserializeDetectsCorruptedValues) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_from_dense(csr, dense.data(), 3, 3), static_cast<int>(SpMVError::SUCCESS));

    std::string path = getTempFilePath("csr_corrupt.bin");
    ASSERT_EQ(csr_serialize(csr, path.c_str()), static_cast<int>(SpMVError::SUCCESS));

    // Values payload starts after magic(4) + version(4) + dims(12).
    FILE* f = std::fopen(path.c_str(), "r+b");
    ASSERT_NE(f, nullptr);
    ASSERT_EQ(std::fseek(f, 20, SEEK_SET), 0);
    int byte = std::fgetc(f);
    ASSERT_NE(byte, EOF);
    ASSERT_EQ(std::fseek(f, 20, SEEK_SET), 0);
    std::fputc(byte ^ 0xFF, f);
    std::fclose(f);

    CSRMatrix* loaded = csr_create(0, 0, 0);
    EXPECT_EQ(csr_deserialize(loaded, path.c_str()), static_cast<int>(SpMVError::FILE_IO));

    csr_destroy(csr);
    csr_destroy(loaded);
    std::remove(path.c_str());
}

// Regression: headers claiming more data than the file contains must be
// rejected before any large allocation happens.
TEST(CSRUnitTest, DeserializeRejectsOversizedHeader) {
    std::string path = getTempFilePath("csr_oversized.bin");
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    uint32_t magic = 0x52534353;
    uint32_t version = 2;
    int rows = 1, cols = 1, nnz = INT_MAX;
    std::fwrite(&magic, sizeof(uint32_t), 1, f);
    std::fwrite(&version, sizeof(uint32_t), 1, f);
    std::fwrite(&rows, sizeof(int), 1, f);
    std::fwrite(&cols, sizeof(int), 1, f);
    std::fwrite(&nnz, sizeof(int), 1, f);
    std::fclose(f);

    CSRMatrix* loaded = csr_create(0, 0, 0);
    EXPECT_EQ(csr_deserialize(loaded, path.c_str()), static_cast<int>(SpMVError::FILE_IO));

    csr_destroy(loaded);
    std::remove(path.c_str());
}

// Regression: rows == INT_MAX would overflow the rows + 1 allocation.
TEST(CSRUnitTest, DeserializeRejectsRowsOverflow) {
    std::string path = getTempFilePath("csr_rowsoverflow.bin");
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    uint32_t magic = 0x52534353;
    uint32_t version = 2;
    int rows = INT_MAX, cols = 1, nnz = 0;
    std::fwrite(&magic, sizeof(uint32_t), 1, f);
    std::fwrite(&version, sizeof(uint32_t), 1, f);
    std::fwrite(&rows, sizeof(int), 1, f);
    std::fwrite(&cols, sizeof(int), 1, f);
    std::fwrite(&nnz, sizeof(int), 1, f);
    std::fclose(f);

    CSRMatrix* loaded = csr_create(0, 0, 0);
    EXPECT_EQ(csr_deserialize(loaded, path.c_str()), static_cast<int>(SpMVError::FILE_IO));

    csr_destroy(loaded);
    std::remove(path.c_str());
}

// Version 1 files (checksum without values) must remain readable.
TEST(CSRUnitTest, DeserializeReadsLegacyVersion1Files) {
    std::string path = getTempFilePath("csr_v1.bin");
    FILE* f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);

    uint32_t magic = 0x52534353;
    uint32_t version = 1;
    int rows = 2, cols = 2, nnz = 2;
    float values[2] = {1.5f, 2.5f};
    int col_indices[2] = {0, 1};
    int row_ptrs[3] = {0, 1, 2};

    std::fwrite(&magic, sizeof(uint32_t), 1, f);
    std::fwrite(&version, sizeof(uint32_t), 1, f);
    std::fwrite(&rows, sizeof(int), 1, f);
    std::fwrite(&cols, sizeof(int), 1, f);
    std::fwrite(&nnz, sizeof(int), 1, f);
    std::fwrite(values, sizeof(float), 2, f);
    std::fwrite(col_indices, sizeof(int), 2, f);
    std::fwrite(row_ptrs, sizeof(int), 3, f);

    // Version 1 checksum: dims + col_indices + row_ptrs (no values).
    uint64_t checksum = 0;
    checksum += static_cast<uint64_t>(rows) + cols + nnz;
    checksum += static_cast<uint64_t>(col_indices[0]) + col_indices[1];
    checksum += static_cast<uint64_t>(row_ptrs[0]) + row_ptrs[1] + row_ptrs[2];
    std::fwrite(&checksum, sizeof(uint64_t), 1, f);
    std::fclose(f);

    CSRMatrix* loaded = csr_create(0, 0, 0);
    ASSERT_EQ(csr_deserialize(loaded, path.c_str()), static_cast<int>(SpMVError::SUCCESS));
    EXPECT_EQ(loaded->num_rows, 2);
    EXPECT_EQ(loaded->nnz, 2);
    EXPECT_FLOAT_EQ(loaded->values[0], 1.5f);
    EXPECT_FLOAT_EQ(loaded->values[1], 2.5f);

    csr_destroy(loaded);
    std::remove(path.c_str());
}

TEST(CSRUnitTest, CreateRejectsOverflowProneSizes) {
    // rows == INT_MAX would overflow the rows + 1 row_ptrs allocation.
    EXPECT_EQ(csr_create(INT_MAX, 1, 0), nullptr);
}
