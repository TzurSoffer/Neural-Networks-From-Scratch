#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include "Mathlib.hpp"

class Pass {
public:
    static double forward(double val);
    static double backward(double val);
};

class ReLU {
public:
    static double forward(double val);
    static double backward(double val);
};

class LeakyReLU {
public:
    static constexpr double alpha = 0.01;
    static double forward(double val);
    static double backward(double val);
};

class Softmax {
public:
    static std::vector<double> forward(std::vector<double> vals);
    static std::vector<std::vector<double>> backward(const std::vector<double>& vals);
};

class Softmax_batch {
public:
    static std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& vals);
    static std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<double>>& vals);
};

class ProtectedSoftmax {
public:
    static std::vector<double> forward(std::vector<double> vals);
    static std::vector<std::vector<double>> backward(const std::vector<double>& vals);
};

class ProtectedSoftmax_batch {
public:
    static std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& vals);
};