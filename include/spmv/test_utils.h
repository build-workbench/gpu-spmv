#ifndef SPMV_TEST_UTILS_H
#define SPMV_TEST_UTILS_H

#include <random>
#include <vector>
#include <cmath>

namespace spmv {
namespace test {

// 随机数生成器
class RandomGenerator {
public:
    RandomGenerator(unsigned seed = 42) : rng_(seed) {}
    
    int randInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng_);
    }
    
    float randFloat(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng_);
    }
    
    bool randBool(float probability = 0.5f) {
        return randFloat(0.0f, 1.0f) < probability;
    }
    
private:
    std::mt19937 rng_;
};

// 生成随机稠密矩阵
inline std::vector<float> generateRandomDenseMatrix(
    int rows, int cols, float density, RandomGenerator& rng,
    float min_val = -10.0f, float max_val = 10.0f) {
    
    std::vector<float> matrix(rows * cols, 0.0f);
    for (int i = 0; i < rows * cols; i++) {
        if (rng.randBool(density)) {
            matrix[i] = rng.randFloat(min_val, max_val);
        }
    }
    return matrix;
}

// 生成随机向量
inline std::vector<float> generateRandomVector(
    int size, RandomGenerator& rng,
    float min_val = -10.0f, float max_val = 10.0f) {
    
    std::vector<float> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = rng.randFloat(min_val, max_val);
    }
    return vec;
}

// 比较两个浮点数组是否相等
inline bool floatArraysEqual(const float* a, const float* b, int size, 
                             float abs_tol = 1e-6f, float rel_tol = 1e-6f) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(a[i] - b[i]);
        float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
        if (diff > abs_tol && diff > rel_tol * max_val) {
            return false;
        }
    }
    return true;
}

// 比较两个整数数组是否相等
inline bool intArraysEqual(const int* a, const int* b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

} // namespace test
} // namespace spmv

#endif // SPMV_TEST_UTILS_H
