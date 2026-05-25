cmake_minimum_required(VERSION 3.18)
project(gpu_spmv VERSION 1.0.0 LANGUAGES CXX)

include(CheckLanguage)
option(SPMV_REQUIRE_CUDA "Fail configuration when CUDA is unavailable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SPMV_WITH_CUDA OFF)

if(SPMV_REQUIRE_CUDA)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        string(TOLOWER "${CMAKE_CXX_COMPILER}" SPMV_CXX_COMPILER_LOWER)
        if(SPMV_CXX_COMPILER_LOWER MATCHES "(conda|miniconda|miniforge|mambaforge|micromamba)")
            message(FATAL_ERROR
                "CUDA builds with Conda-managed host compilers are unsupported in gpu-spmv. "
                "Use the `cuda-linux` or `cuda-linux-release` preset, or configure with "
                "-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ "
                "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++."
            )
        endif()
    endif()

    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA toolkit with nvcc is required to build gpu-spmv. Set CUDAToolkit_ROOT or ensure nvcc is available on PATH, or configure with -DSPMV_REQUIRE_CUDA=OFF for a CPU-only fallback.")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(SPMV_CUDA_RUNTIME_INCLUDE_DIR
        NAMES cuda_runtime.h
        PATHS
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            /usr/local/cuda/include
            /usr/include
        DOC "Directory containing cuda_runtime.h"
    )
    if(NOT SPMV_CUDA_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "Failed to locate cuda_runtime.h for host C++ compilation.")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(SPMV_WITH_CUDA ON)

    # CUDA 架构设置
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
else()
    message(STATUS "SPMV_REQUIRE_CUDA=OFF; configuring in no-CUDA mode. Building core library and CPU tests only.")
endif()

# ---------- 主库 ----------
set(SPMV_SOURCES
    src/csr_matrix.cpp
    src/ell_matrix.cpp
    src/internal/csr_device.cpp
    src/internal/ell_device.cpp
    src/spmv_cpu.cpp
    src/spmv_context.cpp
    src/internal/kernel_selector.cpp
    src/bandwidth.cpp
)

if(SPMV_WITH_CUDA)
    list(APPEND SPMV_SOURCES
        src/spmv_kernels.cu
    )
else()
    list(APPEND SPMV_SOURCES
        src/no_cuda_stubs.cpp
    )
endif()

add_library(spmv STATIC ${SPMV_SOURCES})
target_include_directories(spmv PUBLIC ${CMAKE_SOURCE_DIR}/include)
if(SPMV_WITH_CUDA)
    target_include_directories(spmv PUBLIC ${SPMV_CUDA_RUNTIME_INCLUDE_DIR})
    target_compile_definitions(spmv PUBLIC SPMV_WITH_CUDA=1)
    target_link_libraries(spmv PUBLIC CUDA::cudart)
    set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    target_compile_definitions(spmv PUBLIC SPMV_WITH_CUDA=0)
endif()

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
    tests/test_architecture.cpp
    tests/test_common.cpp
    tests/test_csr.cpp
    tests/test_ell.cpp
    tests/test_kernel_selector.cpp
    tests/test_no_cuda.cpp
)

if(SPMV_WITH_CUDA)
    list(APPEND TEST_SOURCES
        tests/test_spmv.cu
        tests/test_bandwidth.cu
    )
endif()

add_executable(spmv_tests ${TEST_SOURCES})
target_link_libraries(spmv_tests PRIVATE spmv GTest::gtest_main)
if(SPMV_WITH_CUDA)
    target_compile_definitions(spmv_tests PRIVATE SPMV_WITH_CUDA=1)
    set_target_properties(spmv_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    target_compile_definitions(spmv_tests PRIVATE SPMV_WITH_CUDA=0)
endif()
target_compile_options(spmv_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

include(GoogleTest)
gtest_discover_tests(spmv_tests)

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
