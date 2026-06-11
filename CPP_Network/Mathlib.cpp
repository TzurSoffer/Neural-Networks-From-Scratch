#include <iostream>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <random>

double round(double val, int precision) {
    double multiplier = std::pow(10.0, precision);
    return std::round(val * multiplier) / multiplier;
}

double randomNumber(double minimum, double maximum, int decimals) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(minimum, maximum);

    double x = dist(rng);

    double factor = std::pow(10.0, decimals);
    return std::round(x * factor) / factor;
}


std::vector<double> zeroes(int items) {
    return std::vector<double>(items, 0.0);
}
std::vector<std::vector<double>> zeroes(int rows, int cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

double clip(double val, double minVal, double maxVal) {
    return std::clamp(val, minVal, maxVal);
}

double clipAboveZeroBelowOne(double val) {
    return clip(val, 1e-7, 1.0 - 1e-7);
}

float sum(const std::vector<int>& vals) {
    return std::accumulate(vals.begin(), vals.end(), 0.0f);
}

double sum(const std::vector<double>& vals) {
    return std::accumulate(vals.begin(), vals.end(), 0.0);
}

float mean(const std::vector<int>& vals) {
    return sum(vals) / vals.size();
}

double mean(const std::vector<double>& vals) {
    return sum(vals) / vals.size();
}

int argmax(const std::vector<int>& vals) {
    int max = vals[0];
    int maxIndex = 0;
    for (int i = 1; i<vals.size(); i++) {
        if (vals[i] > max) {
            max = vals[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
int argmax(const std::vector<double>& vals) {
    double max = vals[0];
    int maxIndex = 0;
    for (int i = 1; i<vals.size(); i++) {
        if (vals[i] > max) {
            max = vals[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int argmin(const std::vector<int>& vals) {
    int min = vals[0];
    int minIndex = 0;
    for (int i = 1; i<vals.size(); i++) {
        if (vals[i] < min) {
            min = vals[i];
            minIndex = i;
        }
    }
    return minIndex;
}
int argmin(const std::vector<double>& vals) {
    double min = vals[0];
    int minIndex = 0;
    for (int i = 1; i<vals.size(); i++) {
        if (vals[i] < min) {
            min = vals[i];
            minIndex = i;
        }
    }
    return minIndex;
}

double dot2Vectors(const std::vector<double>& vector1,
                   const std::vector<double>& vector2) {
    if (vector1.size() != vector2.size()) {
        throw std::invalid_argument("Vector sizes differ");
    }

    double out = 0.0;

    for (size_t i = 0; i < vector1.size(); i++) {
        out += vector1[i] * vector2[i];
    }

    return out;
}

std::vector<double> dotVectorMatrix(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix) {
    std::vector<double> out = std::vector<double>(matrix.size(), 0.0);
    for (int i = 0; i < matrix.size(); i++) {
            out[i] = dot2Vectors(vector, matrix[i]);
        }
    return out;
}

std::vector<double> addTwoVectors(const std::vector<double>& v1,
                                  const std::vector<double>& v2) {
    std::vector<double> out = std::vector<double>(v1.size(), 0.0);
    for (int i = 0; i<v1.size(); i++) {
        out[i] = v1[i] + v2[i];
    }
    return out;
}

std::vector<std::vector<double>> addTwoMatrices(const std::vector<std::vector<double>>& m1,
                                  const std::vector<std::vector<double>>& m2) {
    std::vector<std::vector<double>> out = std::vector<std::vector<double>>(m1.size(), std::vector<double>(m1[0].size(), 0.0));
    for (int i = 0; i<m1.size(); i++) {
        out[i] = addTwoVectors(m1[i], m2[i]);
    }
    return out;
}

std::vector<double> elementWiseMult(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> out = std::vector<double>(v1.size(), 0.0);
    for (int i = 0; i<v1.size(); i++) {
        out[i] = v1[i] * v2[i];
    }
    return out;
}

std::vector<double> vectorScalerMult(const std::vector<double>& vector, double scale) {
    std::vector<double> out = std::vector<double>(vector.size(), 0.0);
    for (int i = 0; i<vector.size(); i++) {
        out[i] = vector[i] * scale;
    }
    return out;
}

std::vector<std::vector<double>> vectorScalerMult(const std::vector<std::vector<double>>& vector, double scale) {
    std::vector<std::vector<double>> out = std::vector<std::vector<double>>(vector.size(), std::vector<double>(vector[0].size(), 0.0));
    for (int i = 0; i<vector.size(); i++) {
        out[i] = vectorScalerMult(vector[i], scale);
    }
    return out;
}

std::vector<double> scale(const std::vector<double>& vector, double scale) {
    return vectorScalerMult(vector, scale);
}

std::vector<double> normalize(const std::vector<double>& values) {
    std::vector<double> out = std::vector<double>(values.size(), 0.0);
    double total = sum(values);
    for (int i = 0; i<values.size(); i++) {
        out[i] = values[i] / total;
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

std::vector<double> hilbertFlatten(const std::vector<std::vector<double>>& array2d) {
    int n = array2d.size(); // Note size and rowSize must be the same and a power of 2 (this needs to be a square matrix)
    int order = getOrder(n);

    std::vector<double> out = std::vector<double>(n*n, 0.0);
    for (int i=0; i < n * n; i++) {
        std::vector<int> indices = hilbertIndexToXy(i, order);
        out[i] = array2d[indices[1]][indices[0]];
    }

    return out;
}

std::vector<std::vector<double>> hilbertUnflatten(const std::vector<double>& data) {
    int n = static_cast<int>(std::sqrt(data.size()));    // Note data size must be a power of 4
    int order = getOrder(n);

    std::vector<std::vector<double>> array2d = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    for (int i=0; i<data.size(); i++) {
        std::vector<int> indices = hilbertIndexToXy(i, order);
        array2d[indices[1]][indices[0]] = data[i];
    }
    return array2d;
}
