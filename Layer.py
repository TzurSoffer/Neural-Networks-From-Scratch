try:
    import mathlib as Mathlib   #< c++ version (much faster training). Must be compiled first
except ImportError:
    import Mathlib
import Activation
import math

class Layer:
    """A collection of neurons forming a single layer in the network"""
    def __init__(self, inputCount: int, neuronCount: int,
                 activationFunc: Activation=Activation.Pass,   #< Neuron-level activation function
                 ):
        
        self.inputsLen = inputCount
        self.activationFunc = activationFunc
        self.weights, self.biases = self._createLayerData(inputCount, neuronCount)

        self.out = []
        self.preActivationOut = []
        self.inputs = []
        self.d_inputs = []
        self.d_weights = []
        self.d_biases = []

    def _createLayerData(self, inputCount: int=10, neuronCount: int=5) -> tuple[list[float], list[float]]:
        if self.activationFunc in (Activation.ReLU, Activation.LeakyReLU):
            scale = 2.0 / math.sqrt(inputCount)  #< He/Kaiming initialization, scale by 2/sqrt(inputCount) to prevent overflow (better for ReLu activation functions)
        else:
            scale = 1.0 / math.sqrt(inputCount)  #< Xavier/Glorot initialization, scale by 1/sqrt(inputCount) to prevent overflow (better for sigmoid activation functions)
        weights = [[Mathlib.randomNumber(-scale, scale, 2) for _ in range(inputCount)] for _ in range(neuronCount)]
        biases = [0.0 for _ in range(neuronCount)]  # Initialize biases to 0
        return(weights, biases)

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Compute the neurons' output.
        Multiply each input by its corresponding weight, sum the
        results, add the bias and then apply the activation function.
        
        multiply the inputs [i1, i2.. in] with weights [w1, w1... wn] to get i1*w1+i2*w2... +in*wn, then add the bias
        """
        self.inputs = inputs
        if len(self.weights[0]) != len(self.inputs):
            raise Exception("inputs and weight must have the same length!")
        self.preActivationOut = [Mathlib.dot2Vectors(self.inputs, weights)+bias for weights, bias in zip(self.weights, self.biases)]     #< sum(w*x)+b
        self.out = [self.activationFunc.forward(out) for out in self.preActivationOut]
        return(self.out)

    def backward(self, d_values:list[float], inputs=None, preActivationOut=None) -> list[float]:
        """ Compute gradient 
        The following is the explanation for the backward for a SINGLE neuron, a layer would just be a list of these.
            since forward is computed as Activation(sum(x1*w1, x2*w2..., bias)),
            the backward for the weights would be computed as [Activation`(sum(...))*sum`(...)*xi] for every element,
            and the backward for the inputs would be computed as [Activation`(sum(...))*sum`(...)*wi] for every element,
            note that sum`(...) is equal to 1 no matter the reference or input.
            this makes backward for the weights be simplified to [Activation`(sum(...))*xi] for every element,
            and the backward for the inputs be [Activation`(sum(...))*wi] for every element.
        """
        if inputs == None:
            inputs = self.inputs
        if preActivationOut == None:
            preActivationOut = self.preActivationOut
        self.d_inputs = Mathlib.zeroes(len(inputs))
        self.d_weights = []
        self.d_biases = []
        for out, weights, d_val in zip(preActivationOut, self.weights, d_values):
            activation_dx = self.activationFunc.backward(out)*d_val      #< chain rule
            self.d_inputs = Mathlib.addTwoVectors([activation_dx*w for w in weights], self.d_inputs)   #< Total derivative is the sum of the derivatives with respect to the input of each neuron since they are all the same inputs
            self.d_weights.append([activation_dx*i for i in inputs])
            self.d_biases.append(activation_dx)

        return(self.d_inputs)

    def getWeights(self) -> list[list[float]]:
        """ return a list of lists of all weights """
        return(self.weights)
    def setWeights(self, weights: list[list[float]]) -> None:
        self.weights = weights
    def addToWeight(self, indexRow: int, indexCol: int, val: float) -> None:
        self.weights[indexRow][indexCol] = val

    def getBiases(self) -> list[float]:
        """ return a list of all biases """
        return(self.biases)
    def setBiases(self, biases: list[float]) -> None:
        self.biases = biases
    def addToBias(self, index: int, val: float) -> None:
        self.biases[index] = val

    def getOutputs(self) -> list[float]:
        return(self.out)
    def getPreActivationOutputs(self) -> list[float]:
        return(self.preActivationOut)

    def getInputGradient(self) -> list[float]:
        return(self.d_inputs)
    def getWeightsGradient(self) -> list[list[float]]:
        return(self.d_inputs)
    def getBiasesGradient(self) -> list[float]:
        return(self.d_inputs)