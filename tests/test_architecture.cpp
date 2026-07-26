#include "spmv/spmv.h"

#include <gtest/gtest.h>

using namespace spmv;

TEST(SpMVArchitectureTest, ConfigIsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<SpMVConfig>::value);
}

TEST(SpMVArchitectureTest, ResultIsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<SpMVResult>::value);
}

TEST(SpMVArchitectureTest, ThresholdsIsTriviallyCopyable) {
    EXPECT_TRUE(std::is_trivially_copyable<SpMVThresholds>::value);
}
