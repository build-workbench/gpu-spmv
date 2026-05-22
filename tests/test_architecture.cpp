#include "spmv/spmv.h"

#include <gtest/gtest.h>
#include <type_traits>

using namespace spmv;

namespace {

template <typename T, typename = void>
struct HasPublicPrepareTexture : std::false_type {};

template <typename T>
struct HasPublicPrepareTexture<T, std::void_t<decltype(&T::prepare_texture)>> : std::true_type {};

}  // namespace

TEST(SpMVExecutionContextArchitectureTest, PrepareTextureStaysInternal) {
    EXPECT_FALSE(HasPublicPrepareTexture<SpMVExecutionContext>::value);
}
