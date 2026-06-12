#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

#include "Types.hpp"
#include "Mathlib.hpp"
#include "Activation.hpp"

class Layer {
private:
    int inputCount, neuronCount;
    
    std::vector<std::vector<double>> weights, d_weights;
    std::vector<double> biases, d_biases;
    std::vector<double> inputs, d_inputs, out, d_out, preActivationOut;

    ActivationForwardFunc forwardCallback;
    ActivationBackwardFunc backwardCallback;

    void _setActivation(ActivationType type);
    void _createWeights();
    void _createBiases();
    void _createOutputs();

public:
    Layer(int inputCount, int neuronCount, ActivationType type);

    std::vector<double> forward(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& d_values);

    std::vector<std::vector<double>> getWeights();
    std::vector<double> getBiases();
    std::vector<double> getInputGradient();
};