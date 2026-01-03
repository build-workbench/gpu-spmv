#include <gtest/gtest.h>
#include "spmv/common.h"
#include "spmv/cuda_buffer.h"

using namespace spmv;

// 测试错误码字符串转换
TEST(CommonTest, ErrorStringConversion) {
    EXPECT_STREQ(spmv_error_string(SpMVError::SUCCESS), "Success");
    EXPECT_STREQ(spmv_error_string(SpMVError::INVALID_DIMENSION), "Invalid matrix/vector dimension");
    EXPECT_STREQ(spmv_error_string(SpMVError::CUDA_MALLOC), "CUDA memory allocation failed");
    EXPECT_STREQ(spmv_error_string(SpMVError::CUDA_MEMCPY), "CUDA memory copy failed");
    EXPECT_STREQ(spmv_error_string(SpMVError::KERNEL_LAUNCH), "CUDA kernel launch failed");
    EXPECT_STREQ(spmv_error_string(SpMVError::INVALID_FORMAT), "Invalid sparse matrix format");
    EXPECT_STREQ(spmv_error_string(SpMVError::FILE_IO), "File I/O error");
    EXPECT_STREQ(spmv_error_string(SpMVError::OUT_OF_MEMORY), "Out of memory");
    EXPECT_STREQ(spmv_error_string(SpMVError::INVALID_ARGUMENT), "Invalid argument");
}

// 测试 CudaBuffer 构造和析构
TEST(CudaBufferTest, DefaultConstruction) {
    CudaBuffer<float> buffer;
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST(CudaBufferTest, SizedConstruction) {
    CudaBuffer<float> buffer(100);
    EXPECT_NE(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), 100);
    EXPECT_FALSE(buffer.empty());
}

TEST(CudaBufferTest, ZeroSizeConstruction) {
    CudaBuffer<float> buffer(0);
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
}

TEST(CudaBufferTest, MoveConstruction) {
    CudaBuffer<float> buffer1(100);
    float* ptr = buffer1.get();
    
    CudaBuffer<float> buffer2(std::move(buffer1));
    EXPECT_EQ(buffer2.get(), ptr);
    EXPECT_EQ(buffer2.size(), 100);
    EXPECT_EQ(buffer1.get(), nullptr);
    EXPECT_EQ(buffer1.size(), 0);
}

TEST(CudaBufferTest, MoveAssignment) {
    CudaBuffer<float> buffer1(100);
    CudaBuffer<float> buffer2(50);
    float* ptr = buffer1.get();
    
    buffer2 = std::move(buffer1);
    EXPECT_EQ(buffer2.get(), ptr);
    EXPECT_EQ(buffer2.size(), 100);
    EXPECT_EQ(buffer1.get(), nullptr);
}

TEST(CudaBufferTest, CopyFromHost) {
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    CudaBuffer<float> buffer(5);
    
    EXPECT_NO_THROW(buffer.copyFromHost(host_data.data(), 5));
    
    std::vector<float> result(5);
    EXPECT_NO_THROW(buffer.copyToHost(result.data(), 5));
    
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(result[i], host_data[i]);
    }
}

TEST(CudaBufferTest, Resize) {
    CudaBuffer<float> buffer(100);
    EXPECT_EQ(buffer.size(), 100);
    
    buffer.resize(200);
    EXPECT_EQ(buffer.size(), 200);
    EXPECT_NE(buffer.get(), nullptr);
    
    buffer.resize(0);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.get(), nullptr);
}

TEST(CudaBufferTest, Release) {
    CudaBuffer<float> buffer(100);
    EXPECT_NE(buffer.get(), nullptr);
    
    buffer.release();
    EXPECT_EQ(buffer.get(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
}
