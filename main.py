import numpy as np#< use own later
import mathlib
import activation
np.random.seed(0)

def createSetData():
    inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    weights = np.array([[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]])

    biases = [2.0, 3.0, 0.5]
    print("expected output:")
    print(np.dot(inputs, weights.T) + biases)
    print("-------------------------\n\n\n")
    return(inputs, weights, biases)

def createData(n_inputs, n_neurons, batchSize=1):
    inputs = np.random.randn(batchSize, n_neurons)
    weights = 0.1*np.random.randn(n_inputs, n_neurons)
    biases = np.zeros(n_neurons)
    return(inputs, weights, biases)

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
        out = np.array([neuron.calc() for neuron in self.neurons])
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
        return(np.array([b.calc() for b in self.batch]))

if __name__ == "__main__":
    inputs, weights, biases = createSetData()#n_inputs=3, n_neurons=2, batchSize=1)
    
    batch = Batch(inputs, weights, biases, layerActivation=activation.protectedSoftmax)
    # batch = Batch(inputs, weights, biases, activationFunc=activation.softmax, layerActivation=activation.none)
    print(batch.calc())