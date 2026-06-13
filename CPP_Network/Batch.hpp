#pragma once

#include <vector>
#include <memory>
#include "Layer.hpp"
#include "ActivationTypes.hpp"

class Batch {
/*
A batch is a layer with multiple sets of inputs on the same weights and biases. It processes multiple samples at once in a batch and applies the average of the gradients. This improves speed and reduces noise
*/
public:
    Batch(int batchSize, int inputCount, int neuronCount, ActivationType type);

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs);
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& d_values);

    std::vector<std::vector<double>> getWeights() const { return layer->getWeights(); }
    void setWeights(std::vector<std::vector<double>> weights) const { layer->setWeights(weights); };
    void addToWeight(int indexRow, int indexCol, double val) const { layer->addToWeight(indexRow, indexCol, val); };
    std::vector<double> getBiases() const { return layer->getBiases(); }
    void setBiases(std::vector<double> biases) const { layer->setBiases(biases); };
    void addToBias(int index, double val) const { layer->addToBias(index, val); };
    std::vector<std::vector<double>> getWeightsGradient() const { return d_weights; }
    std::vector<double> getBiasesGradient() const { return d_biases; }
    std::vector<std::vector<double>> getOutputs() const { return out; }
    std::vector<std::vector<double>> getPreActivationOutputs() const { return preActivationOut; }
    std::vector<std::vector<double>> getInputGradient() const { return d_inputs; }

private:
    std::unique_ptr<Layer> layer;
    int neuronCount, inputCount, batchSize;

    std::vector<std::vector<double>> d_weights;
    std::vector<double> d_biases;
    std::vector<std::vector<double>> inputs, d_inputs, out, preActivationOut;
};