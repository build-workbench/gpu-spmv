#include "spmv/market_io.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

#include "test_utils.h"

using namespace spmv;
using namespace spmv::test;

namespace {

void writeFile(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    file << content;
}

}  // namespace

TEST(MarketIOTest, ReadsRealGeneralMatrix) {
    std::string path = getTempFilePath("market_real.mtx");
    writeFile(path,
              "%%MatrixMarket matrix coordinate real general\n"
              "% a comment line\n"
              "3 3 4\n"
              "1 1 1.0\n"
              "1 3 2.0\n"
              "2 2 3.0\n"
              "3 3 5.0\n");

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_read_matrix_market(csr, path.c_str()), static_cast<int>(SpMVError::SUCCESS));

    EXPECT_EQ(csr->num_rows, 3);
    EXPECT_EQ(csr->num_cols, 3);
    EXPECT_EQ(csr->nnz, 4);
    EXPECT_TRUE(csr_validate(csr));
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 2), 2.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 1, 1), 3.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 2, 2), 5.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 2, 0), 0.0f);

    csr_destroy(csr);
    std::remove(path.c_str());
}

TEST(MarketIOTest, ExpandsSymmetricMatrix) {
    std::string path = getTempFilePath("market_sym.mtx");
    writeFile(path,
              "%%MatrixMarket matrix coordinate real symmetric\n"
              "3 3 3\n"
              "1 1 1.0\n"
              "2 1 2.0\n"
              "3 1 3.0\n");

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_read_matrix_market(csr, path.c_str()), static_cast<int>(SpMVError::SUCCESS));

    // Diagonal entry stored once, off-diagonal entries mirrored.
    EXPECT_EQ(csr->nnz, 5);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 1, 0), 2.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 1), 2.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 2, 0), 3.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 2), 3.0f);

    csr_destroy(csr);
    std::remove(path.c_str());
}

TEST(MarketIOTest, ReadsPatternAndIntegerFields) {
    std::string pattern_path = getTempFilePath("market_pattern.mtx");
    writeFile(pattern_path,
              "%%MatrixMarket matrix coordinate pattern general\n"
              "2 2 2\n"
              "1 2\n"
              "2 1\n");

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_read_matrix_market(csr, pattern_path.c_str()),
              static_cast<int>(SpMVError::SUCCESS));
    EXPECT_EQ(csr->nnz, 2);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 1), 1.0f);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 1, 0), 1.0f);
    csr_destroy(csr);
    std::remove(pattern_path.c_str());

    std::string int_path = getTempFilePath("market_int.mtx");
    writeFile(int_path,
              "%%MatrixMarket matrix coordinate integer general\n"
              "2 2 1\n"
              "1 1 7\n");

    csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_read_matrix_market(csr, int_path.c_str()), static_cast<int>(SpMVError::SUCCESS));
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 7.0f);
    csr_destroy(csr);
    std::remove(int_path.c_str());
}

TEST(MarketIOTest, SumsDuplicateEntries) {
    std::string path = getTempFilePath("market_dup.mtx");
    writeFile(path,
              "%%MatrixMarket matrix coordinate real general\n"
              "2 2 3\n"
              "1 1 1.0\n"
              "1 1 2.5\n"
              "2 2 4.0\n");

    CSRMatrix* csr = csr_create(0, 0, 0);
    ASSERT_EQ(csr_read_matrix_market(csr, path.c_str()), static_cast<int>(SpMVError::SUCCESS));
    EXPECT_EQ(csr->nnz, 2);
    EXPECT_FLOAT_EQ(csr_get_element(csr, 0, 0), 3.5f);

    csr_destroy(csr);
    std::remove(path.c_str());
}

TEST(MarketIOTest, RejectsUnsupportedFormats) {
    std::string complex_path = getTempFilePath("market_complex.mtx");
    writeFile(complex_path,
              "%%MatrixMarket matrix coordinate complex general\n"
              "2 2 1\n"
              "1 1 1.0 0.0\n");

    CSRMatrix* csr = csr_create(0, 0, 0);
    EXPECT_EQ(csr_read_matrix_market(csr, complex_path.c_str()),
              static_cast<int>(SpMVError::INVALID_FORMAT));
    std::remove(complex_path.c_str());

    std::string dense_path = getTempFilePath("market_array.mtx");
    writeFile(dense_path,
              "%%MatrixMarket matrix array real general\n"
              "2 2\n"
              "1.0\n"
              "0.0\n"
              "0.0\n"
              "1.0\n");
    EXPECT_EQ(csr_read_matrix_market(csr, dense_path.c_str()),
              static_cast<int>(SpMVError::INVALID_FORMAT));
    std::remove(dense_path.c_str());

    csr_destroy(csr);
}

TEST(MarketIOTest, RejectsMalformedFiles) {
    CSRMatrix* csr = csr_create(0, 0, 0);

    // Missing file.
    EXPECT_EQ(csr_read_matrix_market(csr, "/nonexistent/path/missing.mtx"),
              static_cast<int>(SpMVError::FILE_IO));

    // Entry index out of range.
    std::string bad_index_path = getTempFilePath("market_badidx.mtx");
    writeFile(bad_index_path,
              "%%MatrixMarket matrix coordinate real general\n"
              "2 2 1\n"
              "1 5 1.0\n");
    EXPECT_EQ(csr_read_matrix_market(csr, bad_index_path.c_str()),
              static_cast<int>(SpMVError::FILE_IO));
    std::remove(bad_index_path.c_str());

    // Fewer entries than declared.
    std::string short_path = getTempFilePath("market_short.mtx");
    writeFile(short_path,
              "%%MatrixMarket matrix coordinate real general\n"
              "2 2 3\n"
              "1 1 1.0\n");
    EXPECT_EQ(csr_read_matrix_market(csr, short_path.c_str()),
              static_cast<int>(SpMVError::FILE_IO));
    std::remove(short_path.c_str());

    csr_destroy(csr);
}
