#pragma once

#include <vector>
#include "Activation.hpp"

class Layer {
private:
    int inputCount, neuronCount;

    std::vector<std::vector<double>> weights, d_weights;
    std::vector<double> biases, d_biases;

    std::vector<double> inputs, d_inputs, out, d_out, preActivationOut;

    Activation* activationFunc;

    void _createWeights();
    void _createBiases();

public:
    Layer(int inputCount, int neuronCount, Activation* activationFunc);

    std::vector<double> forward(const std::vector<double>& inputs);

    std::vector<double> backward(std::vector<double> d_values);

    std::vector<std::vector<double>> getWeights();

    std::vector<double> getBiases();

    std::vector<double> getInputGradient();
};