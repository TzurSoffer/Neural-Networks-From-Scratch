#include <cmath>
#include <vector>
#include <stdexcept>

#include "Layer.hpp"
#include "Types.hpp"
#include "Mathlib.hpp"      // My custom mathlib
#include "Activation.hpp"   // My custom Actiavtion Classes

Layer::Layer(int inputCount, int neuronCount, ActivationType type) {
    this->inputCount = inputCount;
    this->neuronCount = neuronCount;

    this->d_inputs = zeroes(this->inputCount);
    this->_setActivation(type);
    this->_createWeights();
    this->_createBiases();
    this->_createOutputs();
}

void Layer::_setActivation(ActivationType type) {
        switch (type) {
            case ActivationType::RELU:
                forwardCallback = &ReLU::forward;
                backwardCallback = &ReLU::backward;
                break;
            case ActivationType::LEAKY_RELU:
                forwardCallback = &LeakyReLU::forward;
                backwardCallback = &LeakyReLU::backward;
                break;
            case ActivationType::PASS:
            default:
                forwardCallback = &Pass::forward;
                backwardCallback = &Pass::backward;
                break;
        }
    }

void Layer::_createWeights() {
    double scale;
    scale = 2.0 / std::sqrt(this->inputCount);  // He/Kaiming initialization, scale by 2/sqrt(inputCount) to prevent overflow (better for ReLu activation functions)

    this->weights = std::vector<std::vector<double>>(this->neuronCount, std::vector<double>(this->inputCount, 0.0));
    this->d_weights = this->weights;
    for (int i = 0; i < this->neuronCount; i++) {
        for (int j=0; j < this->inputCount; j++) {
            this->weights[i][j] = randomNumber(-scale, scale, 2);
        }
    }
}

void Layer::Layer::_createBiases() {
    this->biases = std::vector<double>(this->neuronCount, 0.0);
    this->d_biases = this->biases;
}

void Layer::Layer::_createOutputs() {
    this->out = std::vector<double>(this->neuronCount, 0.0);
    this->preActivationOut = this->out;
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) { // inputs are not mutated
    /*
    Compute the neurons' output.
    Multiply each input by its corresponding weight, sum the
    results, add the bias and then apply the activation function.
    
    multiply the inputs [i1, i2.. in] with weights [w1, w1... wn] to get i1*w1+i2*w2... +in*wn, then add the bias
    */
    this->inputs = inputs;
    if (this->weights[0].size() != this->inputs.size()) {
        throw std::invalid_argument("inputs and weight must have the same length!");
    }
    for (int i=0; i < this->weights.size(); i++ ) {
        this->preActivationOut[i] = dot2Vectors(this->inputs, this->weights[i])+this->biases[i]; // sum(w*x)+b
        this->out[i] = this->forwardCallback(this->preActivationOut[i]);
    }
    return this->out;
}

std::vector<double> Layer::backward(const std::vector<double>& d_values, const std::vector<double>& inputs, const std::vector<double>& preActivationOut) {
    std::fill(this->d_inputs.begin(), this->d_inputs.end(), 0.0);
    for (int n=0; n<this->neuronCount; n++) {
        double activation_dx = this->backwardCallback(preActivationOut[n]) * d_values[n];      // chain rule
        for (int i=0; i<this->inputCount; i++ ) {
            this->d_inputs[i] = activation_dx*this->weights[n][i]+this->d_inputs[i];   // chain rule
            this->d_weights[n][i] = activation_dx*inputs[i]; 
        }
        this->d_biases[n] = activation_dx;
    }

    return d_inputs;
}

std::vector<double> Layer::backward(const std::vector<double>& d_values) {
    /*
    Compute gradient 
    The following is the explanation for the backward for a SINGLE neuron, a layer would just be a list of these.
        since forward is computed as Activation(sum(x1*w1, x2*w2..., bias)),
        the backward for the weights would be computed as [Activation`(sum(...))*sum`(...)*xi] for every element,
        and the backward for the inputs would be computed as [Activation`(sum(...))*sum`(...)*wi] for every element,
        note that sum`(...) is equal to 1 no matter the reference or input.
        this makes backward for the weights be simplified to [Activation`(sum(...))*xi] for every element,
        and the backward for the inputs be [Activation`(sum(...))*wi] for every element.
    */
    return this->backward(d_values, this->inputs, this->preActivationOut);
}

std::vector<std::vector<double>> Layer::getWeights() {
    return this->weights;
}
void Layer::setWeights(std::vector<std::vector<double>> weights) {
    this->weights = weights;
}
void Layer::addToWeight(int indexRow, int indexCol, double val) {
    this->weights[indexRow][indexCol] += val;
}

std::vector<double> Layer::getBiases() {
    return this->biases;
}
void Layer::setBiases(std::vector<double> biases) {
    this->biases = biases;
}
void Layer::addToBias(int index, double val) {
    this->biases[index] += val;
}

std::vector<double> Layer::getOutputs() {
    return this->out;
}

std::vector<double> Layer::getPreActivationOutputs() {
    /* The output of the layer before passing the activation function */
    return this->preActivationOut;
}

std::vector<double> Layer::getInputGradient() {
    return this->d_inputs;
}

std::vector<std::vector<double>> Layer::getWeightsGradient() {
    return this->d_weights;
}

std::vector<double> Layer::getBiasesGradient() {
    return this->d_biases;
}