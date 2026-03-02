import numpy as np

import network
import activation
import loss
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
    weights = 0.01*np.random.randn(n_inputs, n_neurons)
    biases = np.zeros(n_neurons)
    return(inputs, weights, biases)

inputs, weights, biases = createSetData()#n_inputs=3, n_neurons=2, batchSize=1)

batch = network.Batch(inputs, weights, biases, layerActivation=activation.protectedSoftmax)
# batch = Batch(inputs, weights, biases, activationFunc=activation.softmax, layerActivation=activation.none)
out = batch.calc()
print(np.array(out))

print(loss.calcLoss([0.7, 0.2, 0.1], 0))
print(loss.calcLoss([0.5, 0.1, 0.4], 1))
print(loss.calcLoss([0.02, 0.9, 0.08], 1))

print(batch.calcLoss([0,0,0]))