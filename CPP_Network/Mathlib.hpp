#pragma once
#include <vector>

double round(double val, int precision);
double randomNumber(double minimum, double maximum, int decimals);
double clip(double val, double minVal, double maxVal);
double clipAboveZeroBelowOne(double val);

float sum(const std::vector<int>& vals);
double sum(const std::vector<double>& vals);

float mean(const std::vector<int>& vals);
double mean(const std::vector<double>& vals);

int argmax(const std::vector<int>& vals);
int argmax(const std::vector<double>& vals);
int argmin(const std::vector<int>& vals);
int argmin(const std::vector<double>& vals);

std::vector<double> zeroes(int items);
std::vector<std::vector<double>> zeroes(int rows, int cols);

double dot2Vectors(const std::vector<double>& vector1,
                   const std::vector<double>& vector2);

std::vector<double> dotVectorMatrix(const std::vector<double>& vector,
                                    const std::vector<std::vector<double>>& matrix);

std::vector<double> addTwoVectors(const std::vector<double>& v1,
                                  const std::vector<double>& v2);

std::vector<std::vector<double>> addTwoMatrices(const std::vector<std::vector<double>>& m1,
                                  const std::vector<std::vector<double>>& m2);

std::vector<double> elementWiseMult(const std::vector<double>& v1,
                                    const std::vector<double>& v2);

std::vector<double> vectorScalerMult(const std::vector<double>& vector, double scale);

std::vector<std::vector<double>> vectorScalerMult(
    const std::vector<std::vector<double>>& vector, double scale);

std::vector<double> scale(const std::vector<double>& vector, double scale);

std::vector<double> normalize(const std::vector<double>& values);

std::vector<int> hilbertIndexToXy(int index, int order);
int getOrder(int n);

std::vector<double> hilbertFlatten(const std::vector<std::vector<double>>& array2d);

std::vector<std::vector<double>> hilbertUnflatten(const std::vector<double>& data);
