#pragma once
#include <vector>

float round(float val, int precision);
float randomNumber(float minimum, float maximum, int decimals);
float clip(float val, float minVal, float maxVal);
float clipAboveZeroBelowOne(float val);

float sum(const std::vector<int>& vals);
float sum(const std::vector<float>& vals);

float mean(std::vector<int> vals);
float mean(std::vector<float> vals);

int argmax(std::vector<int> vals);
int argmax(std::vector<float> vals);
int argmin(std::vector<int> vals);
int argmin(std::vector<float> vals);

std::vector<float> zeroes(int items);
std::vector<std::vector<float>> zeroes(int rows, int cols);

float dot2Vectors(const std::vector<float>& vector1,
                  const std::vector<float>& vector2);

std::vector<float> dotVectorMatrix(std::vector<float> vector,
                                   std::vector<std::vector<float>> matrix);

std::vector<float> addTwoVectors(const std::vector<float>& v1,
                                 const std::vector<float>& v2);

std::vector<std::vector<float>> addTwoMatrices(const std::vector<std::vector<float>>& m1,
                                 const std::vector<std::vector<float>>& m2);

std::vector<float> elementWiseMult(std::vector<float> v1,
                                   std::vector<float> v2);

std::vector<float> vectorScalerMult(std::vector<float> vector, float scale);

std::vector<std::vector<float>> vectorScalerMult(
    std::vector<std::vector<float>> vector, float scale);

std::vector<float> scale(std::vector<float> vector, float scale);

std::vector<float> normalize(std::vector<float> values);

std::vector<int> hilbertIndexToXy(int index, int order);
int getOrder(int n);

std::vector<float> hilbertFlatten(std::vector<std::vector<float>> array2d);

std::vector<std::vector<float>> hilbertUnflatten(std::vector<float> data);
