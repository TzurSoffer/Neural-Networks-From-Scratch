#include <algorithm>
#include <vector>

#include "Mathlib.hpp"
#include "Activation.hpp"
#include "Layer.hpp"
#include "Activation.hpp"
#include "ActivationTypes.hpp"
#include "Batch.hpp"

Batch::Batch(int batchSize, int inputCount, int neuronCount, ActivationType type) :
    layer(std::make_unique<Layer>(inputCount, neuronCount, type)),
    neuronCount(neuronCount),
    inputCount(inputCount),
    batchSize(batchSize),
    d_weights(neuronCount, std::vector<double>(inputCount, 0.0)),
    d_biases(neuronCount, 0.0),
    out(batchSize, std::vector<double>(neuronCount, 0.0)),
    preActivationOut(batchSize, std::vector<double>(neuronCount, 0.0)),
    d_inputs(batchSize, std::vector<double>(neuronCount, 0.0))
{}

std::vector<std::vector<double>> Batch::forward(const std::vector<std::vector<double>>& inputs) {
    /* Forward all layers in the batch and return their outputs */
    this->inputs = inputs;
    for (int i=0; i<this->batchSize; i++) {
        this->out[i] = this->layer->forward(inputs[i]);
        this->preActivationOut[i] = this->layer->getPreActivationOutputs();
    }
    return this->out;
}

std::vector<std::vector<double>> Batch::backward(const std::vector<std::vector<double>>& d_values) {
    /* Average the gradients for the batch for more details, look at the docstring of Layer */

    for (std::vector<double>& row : this->d_weights) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    std::fill(this->d_biases.begin(), this->d_biases.end(), 0.0);

    for (int i=0; i<this->batchSize; i++) {
        this->d_inputs[i] = (this->layer->backward(d_values[i], this->inputs[i], this->preActivationOut[i]));

        this->d_weights = addTwoMatrices(this->d_weights, this->layer->getWeightsGradient());
        this->d_biases = addTwoVectors(this->d_biases, this->layer->getBiasesGradient());
    // AVG is handled at the Loss function (more efficient)
    // scaleFactor = 1 / len(this->inputs)
    // this->d_weights = scale(this->d_weights, scaleFactor)
    // this->d_biases = scale(this->d_biases, scaleFactor)
    }
    return this->d_inputs;
}