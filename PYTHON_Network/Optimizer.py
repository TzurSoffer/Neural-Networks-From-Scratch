from PYTHON_Network.Layer import Layer

class Optimizer_SGD:
    """ Stochastic Gradient Descent optimizer changes the weights and biases based on the gradient descent (derivative) """
    def __init__(self, learning_rate: float=0.01):
        self.learning_rate = learning_rate

    def update(self, layer:Layer) -> None:
        d_weights = layer.getWeightsGradient()
        d_biases = layer.getBiasesGradient()
        for n in range(len(d_weights)):
            for i in range(len(d_weights[n])):
                weightDelta = -self.learning_rate * d_weights[n][i]
                layer.addToWeight(n, i, weightDelta)
            biasDelta = -self.learning_rate * d_biases[n]
            layer.addToBias(n, biasDelta)