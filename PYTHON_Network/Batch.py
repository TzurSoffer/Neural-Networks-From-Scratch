import PYTHON_Network.Mathlib as Mathlib
import PYTHON_Network.Activation as Activation
from PYTHON_Network.ActivationTypes import ActivationType
from PYTHON_Network.Layer import Layer

class Batch():
    """ A batch is a layer with multiple sets of inputs on the same weights and biases. It processes multiple samples at once in a batch and applies the average of the gradients. This improves speed and reduces noise"""
    def __init__(self,
                 batchSize: int,                            #< used for compatibility with C++, but calculated dynamically in python (can safely be ignored here)
                 inputCount: int,
                 neuronCount: int,
                 type: Activation=ActivationType.PASS #< Neuron-level activation function
                 ):
        self.batchSize = batchSize
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self.layer = Layer(inputCount, neuronCount, type)
        
        self.out = []
        self.preActivationOut = []
        self.inputs = []
        self.d_inputs = []
        self.d_weights = []
        self.d_biases = []

    def forward(self, inputs) -> list[list[float]]:
        """ Forward all layers in the batch and return their outputs """
        self.out = []
        self.preActivationOut = []
        self.inputs = inputs
        for i in inputs:
            self.out.append(self.layer.forward(i))
            self.preActivationOut.append(self.layer.getPreActivationOutputs())
        return(self.out)

    def backward(self, d_values: list[list[float]]) -> list[list[float]]:
        """ Average the gradients for the batch for more details, look at the docstring of Layer"""

        self.d_inputs = []
        self.d_weights = Mathlib.zeroes(self.neuronCount, self.inputCount)
        self.d_biases = Mathlib.zeroes(self.neuronCount)

        for inputs, current_d_values, preActivationOut_current in zip(self.inputs, d_values, self.preActivationOut):
            self.d_inputs.append(self.layer.backward(current_d_values, inputs, preActivationOut=preActivationOut_current))

            self.d_weights = Mathlib.addTwoMatrices(self.d_weights, self.layer.getWeightsGradient())  #< self.d_weights changed because we called super
            self.d_biases = Mathlib.addTwoVectors(self.d_biases, self.layer.getBiasesGradient())      #< self.d_biases changed because we called super

        # AVG is handled at the Loss function (more efficient)
        # scaleFactor = 1 / len(self.inputs)
        # self.d_weights = Mathlib.scale(self.d_weights, scaleFactor)
        # self.d_biases = Mathlib.scale(self.d_biases, scaleFactor)

        return(self.d_inputs)

    def getWeights(self) -> list[list[float]]:
        """ return a list of lists of all weights """
        return(self.layer.getWeights())
    def setWeights(self, biases: list[list[float]]) -> None:
        self.layer.setWeights(biases)
    def addToWeight(self, indexRow: int, indexCol: int, val: float) -> None:
        self.layer.addToWeight(indexRow, indexCol, val)

    def getBiases(self) -> list[float]:
        """ return a list of all biases """
        return(self.layer.biases)
    def setBiases(self, biases: list[float]) -> None:
        self.layer.setBiases(biases)
    def addToBias(self, index: int, val: float) -> None:
        self.layer.addToBias(index, val)

    def getOutputs(self) -> list[list[float]]:
        return(self.out)
    def getPreActivationOutputs(self) -> list[list[float]]:
        return(self.preActivationOut)
    def getInputGradient(self) -> list[list[float]]:
        return(self.d_inputs)
    def getWeightsGradient(self) -> list[list[float]]:
        return(self.d_weights)
    def getBiasesGradient(self) -> list[float]:
        return(self.d_biases)