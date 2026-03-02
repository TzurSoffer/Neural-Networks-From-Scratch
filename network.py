import mathlib
import activation
import loss

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
        self.out = None
        self.normalize = normalize
        self.layerActivation = layerActivation
        self.neurons = self._createNeurons(inputs, weights, biases, *args, **kwargs)

    def _createNeurons(self, inputs, weights, biases, *args, **kwargs):
        neurons = []
        for w, b in zip(weights, biases):
            neurons.append(Neuron(inputs, w, b, *args, **kwargs))
        return(neurons)
    
    def calc(self):
        self.out = [neuron.calc() for neuron in self.neurons]
        self.out = self.layerActivation(self.out)
        if self.normalize:
            self.out = mathlib.normalize(self.out)
        return(self.out)

    def calcLoss(self, correctIndex):
        return(loss.calcLoss(self.out, correctIndex))

class Batch:
    def __init__(self, inputsBatch, *args, **kwargs):
        self.batch = self._createBatch(inputsBatch, *args, **kwargs)

    def _createBatch(self, inputsBatch, *args, **kwargs):
        batch = []
        for inputs in inputsBatch:
            batch.append(Layer(inputs, *args, **kwargs))
        return(batch)
    
    def calc(self):
        return([layer.calc() for layer in self.batch])
    
    def calcLoss(self, correctIndexes):
        return(mathlib.mean([self.batch[i].calcLoss(correctIndexes[i]) for i in range(len(self.batch))]))