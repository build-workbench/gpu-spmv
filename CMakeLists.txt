cmake_minimum_required(VERSION 3.18)
project(gpu_spmv VERSION 1.0.0 LANGUAGES CXX)

include(CheckLanguage)
option(SPMV_REQUIRE_CUDA "Fail configuration when CUDA is unavailable" ON)
option(SPMV_BUILD_TESTS "Build unit tests" ON)
option(SPMV_BUILD_EXAMPLES "Build example programs" ON)
option(SPMV_BUILD_BENCHMARKS "Build benchmark tools (requires CUDA)" OFF)

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

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
else()
    message(STATUS "SPMV_REQUIRE_CUDA=OFF; configuring in no-CUDA mode.")
endif()

# ---------- Library ----------
set(SPMV_SOURCES
    src/csr_matrix.cpp
    src/ell_matrix.cpp
    src/internal/csr_device.cpp
    src/internal/ell_device.cpp
    src/spmv_cpu.cpp
    src/internal/kernel_selector.cpp
    src/bandwidth.cpp
    src/market_io.cpp
)

if(SPMV_WITH_CUDA)
    list(APPEND SPMV_SOURCES src/spmv_kernels.cu)
else()
    list(APPEND SPMV_SOURCES src/no_cuda_stubs.cpp)
endif()

add_library(spmv STATIC ${SPMV_SOURCES})
add_library(gpu_spmv::spmv ALIAS spmv)

target_include_directories(spmv PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(SPMV_WITH_CUDA)
    target_include_directories(spmv PUBLIC ${SPMV_CUDA_RUNTIME_INCLUDE_DIR})
    target_compile_definitions(spmv PUBLIC SPMV_WITH_CUDA=1)
    target_link_libraries(spmv PUBLIC CUDA::cudart)
else()
    target_compile_definitions(spmv PUBLIC SPMV_WITH_CUDA=0)
endif()

target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)

# ---------- Tests ----------
if(SPMV_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    set(TEST_SOURCES
        tests/test_architecture.cpp
        tests/test_common.cpp
        tests/test_csr.cpp
        tests/test_ell.cpp
        tests/test_kernel_selector.cpp
        tests/test_market.cpp
        tests/test_no_cuda.cpp
    )

    if(SPMV_WITH_CUDA)
        list(APPEND TEST_SOURCES
            tests/test_spmv.cu
            tests/test_bandwidth.cu
        )
    endif()

    add_executable(spmv_tests ${TEST_SOURCES})
    target_include_directories(spmv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(spmv_tests PRIVATE spmv GTest::gtest_main)
    target_compile_options(spmv_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )

    include(GoogleTest)
    gtest_discover_tests(spmv_tests)
endif()

# ---------- Examples ----------
if(SPMV_BUILD_EXAMPLES)
    add_executable(basic_spmv examples/basic_spmv.cpp)
    target_link_libraries(basic_spmv PRIVATE spmv)
endif()

# ---------- Benchmarks ----------
if(SPMV_BUILD_BENCHMARKS)
    if(SPMV_WITH_CUDA)
        add_executable(spmv_bench tools/spmv_bench.cu)
        target_link_libraries(spmv_bench PRIVATE spmv)
        target_compile_options(spmv_bench PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
    else()
        message(STATUS "SPMV_BUILD_BENCHMARKS=ON requires CUDA; skipping spmv_bench.")
    endif()
endif()

# ---------- Installation ----------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS spmv
    EXPORT gpu_spmvTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/spmv
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "test_utils.h" EXCLUDE
)

install(EXPORT gpu_spmvTargets
    FILE gpu_spmvTargets.cmake
    NAMESPACE gpu_spmv::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpu_spmv
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_spmvConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_spmvConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpu_spmv
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_spmvConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_spmvConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_spmvConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpu_spmv
)
