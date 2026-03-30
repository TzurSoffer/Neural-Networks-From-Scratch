import Mathlib
import Activation
from Neuron import Neuron

class Layer:
    """A collection of neurons forming a single layer in the network"""
    def __init__(self, inputCount: int, neuronCount: int,
                 activationFunc: Activation=Activation.Pass,   #< Neuron-level activation function
                 ):
        
        self.inputsLen = inputCount
        weights, biases = self._createLayerData(inputCount, neuronCount)
        self.activationFunc = activationFunc
        self._createNeurons(weights, biases, activationFunc=activationFunc)  # create neuron objects

    def _createLayerData(self, inputCount: int=10, neuronCount: int=5) -> tuple[list[float], list[float]]:
        weights = [[Mathlib.randomNumber(minimum=-1.0, maximum=1.0) for _ in range(inputCount)] for _ in range(neuronCount)]
        biases = [Mathlib.randomNumber() for _ in range(neuronCount)]
        return(weights, biases)

    def _createNeurons(self, weights: list[float], biases: list[float], activationFunc: Activation) ->  None:
        self.neurons = []
        for w, b in zip(weights, biases):
            self.neurons.append(Neuron(w, b, activationFunc))

    def forward(self, inputs: list[float]) -> list[float]:
        """Calculate outputs for all neurons in the layer """
        return([neuron.forward(inputs) for neuron in self.neurons])
    
    def backward(self, d_values: list[float]) -> list[float]:
        """Calculate gradients for all neurons in the layer."""
        d_inputs = [0]*self.inputsLen
        for neuron, d_val in zip(self.neurons, d_values):
            res = neuron.backward(d_val)
            d_inputs = Mathlib.addTwoVectors(res, d_inputs)   #< Total derivative is the sum of the derivative with respect to the input of each neuron
        return(d_inputs)


    def getWeights(self) -> list[list[float]]:
        """ return a list of lists of all weights """
        return([n.weights for n in self.neurons])
    def getBiases(self) -> list[float]:
        """ return a list of all biases """
        return([n.bias for n in self.neurons])
