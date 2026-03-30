import Mathlib
import Activation
from Layer import Layer

class Batch(Layer):
    """ A batch is a layer with multiple sets of inputs on the same weights and biases. It processes multiple samples at once in a batch and applies the average of the gradients. This improves speed and reduces noise"""
    def __init__(self, batchSize: int, inputCount: int, neuronCount: int,
                 activationFunc: Activation=Activation.Pass          #< Neuron-level activation function
                 ):

        self.batchSize = batchSize
        super().__init__(inputCount, neuronCount, activationFunc)

    def forward(self, inputs) -> list[list[float]]:
        """ Forward all layers in the batch and return their outputs """
        return([super().forward(i) for i in inputs])

    def backward(self, d_valuesBatched: list[list[float]]) -> list[list[float]]:
        """ Average the gradients for all neurons in the batch """
        for neuron in self.neurons:
            neuron.resetDWB()        #< can be removed if properly used with optimizer

        d_inputsBatched = []
        for d_values in d_valuesBatched:
            ## regular layer pass with neuron.backward_batch instead of the normal neuron.backward
            d_inputs = [0]*self.inputsLen
            for neuron, d_val in zip(self.neurons, d_values):
                res = neuron.backward_batch(d_val)
                d_inputs = Mathlib.addTwoVectors(res, d_inputs)   #< Total derivative is the sum of the derivative with respect to the input of each neuron
            d_inputsBatched.append(d_inputs)
        return(d_inputsBatched)
            