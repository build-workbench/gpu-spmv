#include <gtest/gtest.h>
#include "spmv/benchmark.h"
#include "spmv/csr_matrix.h"
#include "spmv/test_utils.h"

using namespace spmv;
using namespace spmv::test;

class BenchmarkPropertyTest : public ::testing::Test {
protected:
    RandomGenerator rng{42};
    static constexpr int NUM_ITERATIONS = 50;  // 减少迭代次数以加快测试
};

// **Feature: spmv-gpu, Property 13: Benchmark Metrics Completeness**
// **Validates: Requirements 6.1, 6.3**
TEST_F(BenchmarkPropertyTest, MetricsCompleteness) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(10, 100);
        int cols = rng.randInt(10, 100);
        float density = rng.randFloat(0.05f, 0.3f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        auto x = generateRandomVector(cols, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        csr_to_gpu(csr);
        
        BenchmarkConfig bench_config;
        bench_config.num_warmup_runs = 2;
        bench_config.num_runs = 5;
        
        BenchmarkResult result = benchmark_csr(csr, x.data(), nullptr, &bench_config);
        
        // 验证所有度量都有效
        EXPECT_GT(result.execution_time_ms, 0.0f)
            << "Execution time should be positive at iteration " << iter;
        
        EXPECT_GE(result.gflops, 0.0f)
            << "GFLOPS should be non-negative at iteration " << iter;
        
        EXPECT_GE(result.bandwidth_gb_s, 0.0f)
            << "Bandwidth should be non-negative at iteration " << iter;
        
        // 验证统计度量
        EXPECT_LE(result.min_time_ms, result.avg_time_ms)
            << "Min should be <= avg at iteration " << iter;
        
        EXPECT_LE(result.avg_time_ms, result.max_time_ms)
            << "Avg should be <= max at iteration " << iter;
        
        EXPECT_GE(result.stddev_time_ms, 0.0f)
            << "Stddev should be non-negative at iteration " << iter;
        
        EXPECT_EQ(result.num_runs, bench_config.num_runs)
            << "Num runs mismatch at iteration " << iter;
        
        csr_destroy(csr);
    }
}

// **Feature: spmv-gpu, Property 14: Benchmark JSON Round Trip**
// **Validates: Requirements 6.5**
TEST_F(BenchmarkPropertyTest, JSONRoundTrip) {
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        int rows = rng.randInt(10, 100);
        int cols = rng.randInt(10, 100);
        float density = rng.randFloat(0.05f, 0.3f);
        
        auto dense = generateRandomDenseMatrix(rows, cols, density, rng);
        auto x = generateRandomVector(cols, rng);
        
        CSRMatrix* csr = csr_create(0, 0, 0);
        csr_from_dense(csr, dense.data(), rows, cols);
        csr_to_gpu(csr);
        
        BenchmarkConfig bench_config;
        bench_config.num_warmup_runs = 2;
        bench_config.num_runs = 5;
        
        BenchmarkResult original = benchmark_csr(csr, x.data(), nullptr, &bench_config);
        
        // 序列化到 JSON
        std::string json = benchmark_to_json(original);
        EXPECT_FALSE(json.empty()) << "JSON should not be empty";
        
        // 反序列化
        BenchmarkResult loaded = benchmark_from_json(json);
        
        // 验证数据一致性
        EXPECT_FLOAT_EQ(original.execution_time_ms, loaded.execution_time_ms);
        EXPECT_FLOAT_EQ(original.gflops, loaded.gflops);
        EXPECT_FLOAT_EQ(original.bandwidth_gb_s, loaded.bandwidth_gb_s);
        EXPECT_FLOAT_EQ(original.avg_time_ms, loaded.avg_time_ms);
        EXPECT_FLOAT_EQ(original.min_time_ms, loaded.min_time_ms);
        EXPECT_FLOAT_EQ(original.max_time_ms, loaded.max_time_ms);
        EXPECT_FLOAT_EQ(original.stddev_time_ms, loaded.stddev_time_ms);
        EXPECT_EQ(original.num_runs, loaded.num_runs);
        
        csr_destroy(csr);
    }
}

// 单元测试
TEST(BenchmarkUnitTest, BasicBenchmark) {
    std::vector<float> dense = {1, 0, 2, 0, 3, 4, 0, 0, 5};
    std::vector<float> x = {1, 1, 1};
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 3, 3);
    csr_to_gpu(csr);
    
    BenchmarkConfig config;
    config.num_warmup_runs = 1;
    config.num_runs = 3;
    
    BenchmarkResult result = benchmark_csr(csr, x.data(), nullptr, &config);
    
    EXPECT_GT(result.execution_time_ms, 0.0f);
    EXPECT_EQ(result.num_runs, 3);
    EXPECT_LE(result.min_time_ms, result.max_time_ms);
    
    csr_destroy(csr);
}

TEST(BenchmarkUnitTest, GPUvsCPUComparison) {
    std::vector<float> dense(100, 0.0f);
    for (int i = 0; i < 100; i += 2) {
        dense[i] = 1.0f;
    }
    std::vector<float> x(10, 1.0f);
    
    CSRMatrix* csr = csr_create(0, 0, 0);
    csr_from_dense(csr, dense.data(), 10, 10);
    csr_to_gpu(csr);
    
    BenchmarkConfig config;
    config.num_warmup_runs = 1;
    config.num_runs = 3;
    
    ComparisonResult comp = compare_gpu_cpu_csr(csr, x.data(), nullptr, &config);
    
    EXPECT_GT(comp.gpu_result.execution_time_ms, 0.0f);
    EXPECT_GT(comp.cpu_result.execution_time_ms, 0.0f);
    EXPECT_GE(comp.speedup, 0.0f);
    
    csr_destroy(csr);
}

TEST(BenchmarkUnitTest, JSONFormat) {
    BenchmarkResult result;
    result.name = "Test";
    result.execution_time_ms = 1.5f;
    result.gflops = 2.5f;
    result.bandwidth_gb_s = 100.0f;
    result.avg_time_ms = 1.5f;
    result.min_time_ms = 1.0f;
    result.max_time_ms = 2.0f;
    result.stddev_time_ms = 0.3f;
    result.num_runs = 10;
    
    std::string json = benchmark_to_json(result);
    
    EXPECT_NE(json.find("\"name\""), std::string::npos);
    EXPECT_NE(json.find("\"execution_time_ms\""), std::string::npos);
    EXPECT_NE(json.find("\"gflops\""), std::string::npos);
    EXPECT_NE(json.find("\"bandwidth_gb_s\""), std::string::npos);
    EXPECT_NE(json.find("\"num_runs\""), std::string::npos);
}
