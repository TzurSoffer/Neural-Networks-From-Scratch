#include <iostream>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>

float round(float val, int precision) {
    float multiplier = std::pow(10, precision);
    return std::round(val*multiplier)/multiplier;
}

float randomNumber(float minimum, float maximum, int decimals) {
    float fraction = static_cast<float>(std::rand()) / RAND_MAX;
    return round(minimum + fraction*(maximum-minimum), decimals);
}

std::vector<float> zeroes(int items) {
    return std::vector<float>(items, 0.0f);
}
std::vector<std::vector<float>> zeroes(int rows, int cols) {
    return std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f));
}

float clip(float val, float minVal, float maxVal) {
    return std::clamp(val, minVal, maxVal);
}

float clipAboveZeroBelowOne(float val) {
    return clip(val, 1e-7, 1-1e-7);
}

float sum(const std::vector<int>& vals) {
    return std::accumulate(vals.begin(), vals.end(), 0.0f);
}

float sum(const std::vector<float>& vals) {
    return std::accumulate(vals.begin(), vals.end(), 0.0f);
}

float mean(std::vector<int> vals) {
    return sum(vals)/vals.size();
}

float mean(std::vector<float> vals) {
    return sum(vals)/vals.size();
}

int argmax(std::vector<int> vals) {
    int max = vals[0];
    int maxIndex = 0;
    for (int i=1; i<vals.size(); i++) {
        if (vals[i] > max) {
            max = vals[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
int argmax(std::vector<float> vals) {
    int max = vals[0];
    int maxIndex = 0;
    for (int i=1; i<vals.size(); i++) {
        if (vals[i] > max) {
            max = vals[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int argmin(std::vector<int> vals) {
    int min = vals[0];
    int minIndex = 0;
    for (int i=1; i<vals.size(); i++) {
        if (vals[i] < min) {
            min = vals[i];
            minIndex = i;
        }
    }
    return minIndex;
}
int argmin(std::vector<float> vals) {
    int min = vals[0];
    int minIndex = 0;
    for (int i=1; i<vals.size(); i++) {
        if (vals[i] < min) {
            min = vals[i];
            minIndex = i;
        }
    }
    return minIndex;
}

float dot2Vectors(const std::vector<float>& vector1,
                  const std::vector<float>& vector2) {
    if (vector1.size() != vector2.size()) {
        throw std::invalid_argument("Vector sizes differ");
    }

    float out = 0.0f;

    for (size_t i = 0; i < vector1.size(); i++) {
        out += vector1[i] * vector2[i];
    }

    return out;
}

std::vector<float> dotVectorMatrix(std::vector<float> vector, std::vector<std::vector<float>> matrix) {
    std::vector<float> out = std::vector<float>(matrix.size(), 0.0f);
    for (int i=0; i<matrix.size(); i++) {
            out[i] = (dot2Vectors(vector, matrix[i]));
        }
    return out;
}

std::vector<float> addTwoVectors(const std::vector<float>& v1,
                                 const std::vector<float>& v2) {
    std::vector<float> out = std::vector<float>(v1.size(), 0.0f);
    for (int i=0; i<v1.size(); i++) {
        out[i] = v1[i]+v2[i];
    }
    return out;
}

std::vector<std::vector<float>> addTwoMatrices(const std::vector<std::vector<float>>& m1,
                                 const std::vector<std::vector<float>>& m2) {
    std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m1.size(), std::vector<float>(m1[0].size(), 0.0f));
    for (int i=0; i<m1.size(); i++) {
        out[i] = addTwoVectors(m1[i], m2[i]);
    }
    return out;
}

std::vector<float> elementWiseMult(std::vector<float> v1, std::vector<float> v2) {
    std::vector<float> out = std::vector<float>(v1.size(), 0.0f);
    for (int i=0; i<v1.size(); i++) {
        out[i] = v1[i]*v2[i];
    }
    return out;
}

std::vector<float> vectorScalerMult(std::vector<float> vector, float scale) {
    std::vector<float> out = std::vector<float>(vector.size(), 0.0f);
    for (int i=0; i<vector.size(); i++) {
        out[i] = vector[i]*scale;
    }
    return out;
}

std::vector<std::vector<float>> vectorScalerMult(std::vector<std::vector<float>> vector, float scale) {
    std::vector<std::vector<float>> out = std::vector<std::vector<float>>(vector.size(), std::vector<float>(vector[0].size(), 0.0f));
    for (int i=0; i<vector.size(); i++) {
        out[i] = vectorScalerMult(vector[i], scale);
    }
    return out;
}

std::vector<float> scale(std::vector<float> vector, float scale) {
    return vectorScalerMult(vector, scale);
}

std::vector<float> normalize(std::vector<float> values) {
    std::vector<float> out = std::vector<float>(values.size(), 0.0f);
    float total = sum(values);
    for (int i=0; i<values.size(); i++) {
        out[i] = values[i]/total;
    }
    return out;
}

std::vector<int> hilbertIndexToXy(int index, int order) {
    int n = 1 << order;
    int x = 0;
    int y = 0;
    int t = index;
    int s = 1;
    int rx;
    int ry;
    int tmp;

    while (s < n){
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);

        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            tmp = x;
            x = y;
            y = tmp;
        }

        x += s * rx;
        y += s * ry;

        t /= 4;
        s *= 2;
    }

    std::vector<int> out = std::vector<int>(2, x);
    out[1] = y;
    return out;
}

int getOrder(int n) {
    if (n <= 0) {
        throw std::invalid_argument("n must be positive");
    }

    return static_cast<int>(std::floor(std::log2(n)));
}

std::vector<float> hilbertFlatten(std::vector<std::vector<float>> array2d) {
    int n = array2d.size(); // Note size and rowSize must be the same and a power of 2 (this needs to be a square matrix)
    int order = getOrder(n);

    std::vector<float> out = std::vector<float>(n*n, 0.0f);
    for (int i=0; i < n * n; i++) {
        std::vector<int> indices = hilbertIndexToXy(i, order);
        out[i] = array2d[indices[1]][indices[0]];
    }

    return out;
}

std::vector<std::vector<float>> hilbertUnflatten(std::vector<float> data) {
    int n = static_cast<int>(std::sqrt(data.size()));    // Note data size must be a power of 4
    int order = getOrder(n);

    std::vector<std::vector<float>> array2d = std::vector<std::vector<float>>(n, std::vector<float>(n, 0.0f));

    for (int i=0; i<data.size(); i++) {
        std::vector<int> indices = hilbertIndexToXy(i, order);
        array2d[indices[1]][indices[0]] = data[i];
    }
    return array2d;
}
