cmake_minimum_required(VERSION 3.18)
project(gpu_spmv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构设置
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
file(GLOB_RECURSE SPMV_SOURCES "src/*.cpp" "src/*.cu")

# 主库
add_library(spmv STATIC ${SPMV_SOURCES})
target_include_directories(spmv PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# 测试可执行文件
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.cu")
add_executable(spmv_tests ${TEST_SOURCES})
target_link_libraries(spmv_tests spmv GTest::gtest_main)
set_target_properties(spmv_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(GoogleTest)
gtest_discover_tests(spmv_tests)

# 基准测试可执行文件
file(GLOB_RECURSE BENCH_SOURCES "benchmarks/*.cpp" "benchmarks/*.cu")
if(BENCH_SOURCES)
    add_executable(spmv_benchmark ${BENCH_SOURCES})
    target_link_libraries(spmv_benchmark spmv)
    set_target_properties(spmv_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
