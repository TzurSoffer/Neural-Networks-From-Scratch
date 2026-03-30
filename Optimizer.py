from Layer import Layer

class Optimizer_SGD:
    """ Stochastic Gradient Descent optimizer changes the weights and biases based on the gradient descent (derivative) """
    def __init__(self, learning_rate: float=0.01):
        self.learning_rate = learning_rate

    def update(self, layer:Layer) -> None:
        for neuron in layer.neurons:
            neuron.weights = [w - self.learning_rate * dw for w, dw in zip(neuron.weights, neuron.d_weights)]
            neuron.bias -= self.learning_rate * neuron.d_bias
            neuron.resetDWB()