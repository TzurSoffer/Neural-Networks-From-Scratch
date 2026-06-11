try:
    import mathlib as Mathlib   #< c++ version (much faster training). Must be compiled first
except ImportError:
    import Mathlib
import Activation
from Layer import Layer

class Batch(Layer):
    """ A batch is a layer with multiple sets of inputs on the same weights and biases. It processes multiple samples at once in a batch and applies the average of the gradients. This improves speed and reduces noise"""
    def __init__(self, inputCount: int, neuronCount: int,
                 activationFunc: Activation=Activation.Pass          #< Neuron-level activation function
                 ):

        super().__init__(inputCount, neuronCount, activationFunc)

    def forward(self, inputs) -> list[list[float]]:
        """ Forward all layers in the batch and return their outputs """
        res = []
        self.perActivationOutBatched = []
        for i in inputs:
            res.append(super().forward(i))
            self.perActivationOutBatched.append(self.perActivationOut)
        self.inputs = inputs
        return(res)

    def backward(self, d_valuesBatched: list[list[float]]) -> list[list[float]]:
        """ Average the gradients for the batch for more details, look at the docstring of Layer"""

        d_inputs = []

        d_weights = Mathlib.zeroes(len(self.weights), len(self.weights[0]))
        d_biases = Mathlib.zeroes(len(self.weights))

        for inputs, d_values, perActivationOut in zip(self.inputs, d_valuesBatched, self.perActivationOutBatched):
            self.perActivationOut = perActivationOut
            d_inputs.append(super().backward(d_values, inputs))

            d_weights = Mathlib.addTwoMatrices(d_weights, self.d_weights)  #< self.d_weights changed because we called super
            d_biases = Mathlib.addTwoVectors(d_biases, self.d_biases)      #< self.d_biases changed because we called super
        self.d_weights = d_weights
        self.d_biases = d_biases

        # AVG is handled at the Loss function (more efficient)
        # scaleFactor = 1 / len(self.inputs)
        # self.d_weights = Mathlib.scale(d_weights, scaleFactor)
        # self.d_biases = Mathlib.scale(d_biases, scaleFactor)

        return(d_inputs)
