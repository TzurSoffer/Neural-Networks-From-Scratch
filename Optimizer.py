from Layer import Layer

class Optimizer_SGD:
    """ Stochastic Gradient Descent optimizer changes the weights and biases based on the gradient descent (derivative) """
    def __init__(self, learning_rate: float=0.01):
        self.learning_rate = learning_rate

    def update(self, layer:Layer) -> None:
        for i in range(len(layer.weights)):
            layer.weights[i] = [w - self.learning_rate * dw for w, dw in zip(layer.weights[i], layer.d_weights[i])]
            layer.biases[i] -= self.learning_rate * layer.d_biases[i]