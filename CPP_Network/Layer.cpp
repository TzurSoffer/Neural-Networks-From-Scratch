#include <cmath>
#include <vector>
#include <stdexcept>

#include "Types.hpp"
#include "Mathlib.hpp"      // My custom mathlib
#include "Activation.hpp"   // My custom Actiavtion Classes

class Layer{
private:
    int inputCount, neuronCount;
    std::vector<std::vector<double>> weights, d_weights;
    std::vector<double> biases, d_biases;
    std::vector<double> inputs, d_inputs, out, d_out, preActivationOut;

    ActivationForwardFunc forwardCallback;
    ActivationBackwardFunc backwardCallback;

    void _setActivation(ActivationType type) {
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

    void _createWeights() {
        double scale;
        scale = 2.0 / std::sqrt(this->inputCount);  // He/Kaiming initialization, scale by 2/sqrt(inputCount) to prevent overflow (better for ReLu activation functions)

        this->weights = std::vector<std::vector<double>>(this->neuronCount, std::vector<double>(this->inputCount, 0.0));
        this->d_weights = std::vector<std::vector<double>>(this->neuronCount, std::vector<double>(this->inputCount, 0.0));
        for (int i = 0; i < this->neuronCount; i++) {
            for (int j=0; j < this->inputCount; j++) {
                this->weights[i][j] = randomNumber(-scale, scale, 2);
            }
        }
    }

    void _createBiases() {
        this->biases = std::vector<double>(this->neuronCount, 0.0);
        this->d_biases = std::vector<double>(this->neuronCount, 0.0);
    }

    void _createOutputs() {
        this->out = std::vector<double>(this->neuronCount, 0.0);
        this->d_out = std::vector<double>(this->neuronCount, 0.0);
        this->preActivationOut = std::vector<double>(this->neuronCount, 0.0);
    }

public:
    /* 
    A collection of neurons forming a single layer in the network
    */
    
    Layer(int inputCount, int neuronCount, ActivationType type) {
        this->inputCount = inputCount;
        this->neuronCount = neuronCount;
        
        this->_setActivation(type);
        this->_createWeights();
        this->_createBiases();
        this->_createOutputs();
    }

    std::vector<double> forward(const std::vector<double>& inputs) { // inputs are not mutated
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

    std::vector<double> backward(const std::vector<double>& d_values) {
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
        this->d_inputs = zeroes(this->inputCount);
        for (int n=0; n<this->neuronCount; n++) {
            double activation_dx = this->backwardCallback(this->preActivationOut[n]) * d_values[n];      // chain rule
            for (int i=0; i<this->inputCount; i++ ) {
                this->d_inputs[i] = activation_dx*this->weights[n][i]+this->d_inputs[i];   // chain rule
                this->d_weights[n][i] = activation_dx*this->inputs[i];
            }
            this->d_biases[n] = activation_dx;
        }

        return d_inputs;
    }

    std::vector<std::vector<double>> getWeights() {
        return this->weights;
    }

    std::vector<double> getBiases() {
        return this->biases;
    }

    std::vector<double> getInputGradient() {
        return this->d_inputs;
    }
};