#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>

#include "ActivationTypes.hpp"
#include "Mathlib.hpp"
#include "Activation.hpp"

class Layer {
private:
    int inputCount, neuronCount;
    
    std::vector<std::vector<double>> weights, d_weights;    // So optimizer and Batch can directly change them
    std::vector<double> biases, d_biases;
    std::vector<double> inputs, d_inputs, out, preActivationOut;

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
    std::vector<double> backward(const std::vector<double>& d_values, const std::vector<double>& inputs, const std::vector<double>& preActivationOut);

    std::vector<std::vector<double>> getWeights();
    void setWeights(std::vector<std::vector<double>> weights);
    void addToWeight(int indexRow, int indexCol, double val);
    std::vector<double> getBiases();
    void setBiases(std::vector<double> biases);
    void addToBias(int index, double val);
    std::vector<std::vector<double>> getWeightsGradient();
    std::vector<double> getOutputs();
    std::vector<double> getPreActivationOutputs();
    std::vector<double> getInputGradient();
    std::vector<double> getBiasesGradient();
};