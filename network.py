import mathlib
import activation

class Neuron:
    def __init__(self, inputs, weights, bias, activationFunc=activation.none):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activationFunc = activationFunc
    
    def calc(self):
        return(self.activationFunc(mathlib.dot2Vectors(self.inputs, self.weights)+self.bias))

class Layer:
    def __init__(self, inputs, weights, biases, normalize=True, layerActivation=activation.none, *args, **kwargs):
        self.normalize = normalize
        self.layerActivation = layerActivation
        self.neurons = self._createNeurons(inputs, weights, biases, *args, **kwargs)

    def _createNeurons(self, inputs, weights, biases, *args, **kwargs):
        neurons = []
        for w, b in zip(weights, biases):
            neurons.append(Neuron(inputs, w, b, *args, **kwargs))
        return(neurons)
    
    def calc(self):
        out = [neuron.calc() for neuron in self.neurons]
        out = self.layerActivation(out)
        if self.normalize:
            out = mathlib.normalize(out)
        return(out)

class Batch:
    def __init__(self, inputsBatch, *args, **kwargs):
        self.batch = self._createBatch(inputsBatch, *args, **kwargs)

    def _createBatch(self, inputsBatch, *args, **kwargs):
        batch = []
        for inputs in inputsBatch:
            batch.append(Layer(inputs, *args, **kwargs))
        return(batch)
    
    def calc(self):
        return([b.calc() for b in self.batch])