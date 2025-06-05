import numpy as np#< use own later

def createData(n_inputs, n_neurons, batchSize=1):
    inputs = np.random.randn(batchSize, n_neurons)
    weights = 0.1*np.random.randn(n_inputs, n_neurons)
    biases = np.zeros((1, n_neurons))
    return(inputs, weights, biases)

class Neuron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
    
    def calc(self):
        return(np.dot(self.inputs, self.weights)+self.bias)

class Layer:
    def __init__(self, inputs, weights, biases):
        self.inputs = inputs
        self.neurons = self._createNeurons(weights, biases)

    def _createNeurons(self, weights, biases):
        neurons = []
        for w, b in zip(weights, biases):
            neurons.append(Neuron(self.inputs, w, b))
        return(neurons)
    
    def calc(self):
        return(np.array([neuron.calc() for neuron in self.neurons]))

class Batch:
    def __init__(self, inputsBatch, weights, biases):
        self.batch = self._createBatch(inputsBatch, weights, biases)

    def _createBatch(self, inputsBatch, weights, biases):
        batch = []
        for inputs in inputsBatch:
            batch.append(Layer(inputs, weights, biases))
        return(batch)
    
    def calc(self):
        return(np.array([b.calc() for b in self.batch]))

if __name__ == "__main__":
    inputs, weights, biases = createData(4, 5, batchSize=1)
    
    batch = Batch(inputs, weights, biases)
    print(batch.calc())