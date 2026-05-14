cmake_minimum_required(VERSION 3.18)
project(gpu_spmv VERSION 1.0.0 LANGUAGES CXX)

include(CheckLanguage)
option(SPMV_REQUIRE_CUDA "Fail configuration when CUDA is unavailable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT SPMV_REQUIRE_CUDA)
    message(STATUS "SPMV_REQUIRE_CUDA=OFF; configuring in no-CUDA mode. No build targets will be generated.")
    enable_testing()
    return()
endif()

check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA toolkit with nvcc is required to build gpu-spmv. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH, or configure with -DSPMV_REQUIRE_CUDA=OFF for a configure-only fallback.")
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构设置
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# ---------- 主库 ----------
set(SPMV_SOURCES
    src/csr_matrix.cpp
    src/ell_matrix.cpp
    src/spmv_cpu.cpp
    src/internal/kernel_selector.cpp
    src/spmv_kernels.cu
    src/bandwidth.cpp
    src/benchmark.cu
    src/pagerank.cu
)

add_library(spmv STATIC ${SPMV_SOURCES})
target_include_directories(spmv PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CUDA 编译选项
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)

# ---------- Google Test ----------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---------- 测试 ----------
set(TEST_SOURCES
    tests/test_common.cpp
    tests/test_csr.cpp
    tests/test_ell.cpp
    tests/test_kernel_selector.cpp
    tests/test_spmv.cu
    tests/test_bandwidth.cu
    tests/test_benchmark.cu
    tests/test_pagerank.cu
)

add_executable(spmv_tests ${TEST_SOURCES})
target_link_libraries(spmv_tests PRIVATE spmv GTest::gtest_main)
set_target_properties(spmv_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(spmv_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include(GoogleTest)
gtest_discover_tests(spmv_tests)

# ---------- 基准测试 ----------
add_executable(spmv_benchmark benchmarks/main.cu)
target_link_libraries(spmv_benchmark PRIVATE spmv)
set_target_properties(spmv_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(spmv_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# ---------- Installation ----------
include(GNUInstallDirs)

install(TARGETS spmv
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/spmv
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
