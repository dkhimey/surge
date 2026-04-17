#pragma once

#include <iostream>
#include <vector>
#include <cmath>

inline void normalizeVector(float* data, float* norm_array, size_t dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++)
        norm += data[i] * data[i];
    norm = 1.0f / (std::sqrt(norm) + 1e-30f);
    for (int i = 0; i < dim; i++)
        norm_array[i] = data[i] * norm;
}

inline bool compareVectors(float* v1, float* v2, size_t dim, float eps = 1e-5f) {
    for (size_t i = 0; i < dim; i++) {
        if (std::abs(v1[i] - v2[i]) > eps) return false;
    }
    return true;
}

inline void printVector(float* v, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

inline float computeInnerProductDistance(float* a, float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return 1.0 - dot;
}

// inline float computeEucleadianDistance(float* a, float* b, size_t dim) {
//     float dist = 0.0f;
//     for (size_t i = 0; i < dim; i++) {
//         dist += (a[i] - b[i]) * (a[i] - b[i]);
//     }
//     return dist;
// }

inline float computeEuclideanDistance(const float* __restrict__ a, const float* __restrict__ b, size_t dim) {
    float dist = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        dist += d * d;
    }
    return dist;
}

inline void logIntVec(std::vector<int>& vec, std::string& output_filename) {
    std::ofstream outFile(output_filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(vec.data()), sizeof(int)*vec.size());
    outFile.close();
}

inline void logFloatVec(std::vector<float>& vec, std::string& output_filename) {
    std::ofstream outFile(output_filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(vec.data()), sizeof(float)*vec.size());
    outFile.close();
}

inline void logDoubleVec(std::vector<double>& vec, std::string& output_filename) {
    std::ofstream outFile(output_filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(vec.data()), sizeof(double)*vec.size());
    outFile.close();
}
