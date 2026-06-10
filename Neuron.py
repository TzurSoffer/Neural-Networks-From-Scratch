import Activation
try:
    import mathlib as Mathlib   #< c++ version (much faster training). Must be compiled first
except ImportError:
    import Mathlib

class Neuron:
    """ A single neuron, stores its inputs, weights, bias, and activation function """
    def __init__(self, weights, bias, activationFunc=Activation.Pass):
        self.inputs = None
        self.weights = weights    #< one weight for every input
        self.bias = bias          #< one bias per neuron

        self.out = None
        self.d_inputs = None
        self.d_weights = [0]*len(self.weights)
        self.d_bias = 0

        self.activationFunc = activationFunc

    def forward(self, inputs: list[float]) -> float:
        """
        Compute the neuron's output.
        Multiply each input by its corresponding weight, sum the
        results, add the bias and then apply the activation function.
        
        multiply the inputs [i1, i2.. in] with weights [w1, w1... wn] to get i1*w1+i2*w2... +in*wn, then add the bias
        """
        self.inputs = inputs
        if len(self.weights) != len(self.inputs):
            raise Exception("inputs and weight must have the same length!")
        self.out = Mathlib.dot2Vectors(self.inputs, self.weights)+self.bias     #< sum(w*x)+b
        return(self.activationFunc.forward(self.out))

    def backward(self, d_val:float) -> list[float]:
        """ Compute gradient 
        since forward is computed as Activation(sum(x1*w1, x2*w2..., bias)),
        the backward for the weights would be computed as [Activation`(sum(...))*sum`(...)*xi] for every element,
        and the backward for the inputs would be computed as [Activation`(sum(...))*sum`(...)*wi] for every element,
        note that sum`(...) is equal to 1 no matter the reference or input.
        this makes backward for the weights be simplified to [Activation`(sum(...))*xi] for every element,
        and the backward for the inputs be [Activation`(sum(...))*wi] for every element.
        """
        activation_dx = self.activationFunc.backward(self.out)*d_val #< chain rule
        self.d_inputs = [activation_dx*w for w in self.weights]
        self.d_weights = [activation_dx*i for i in self.inputs]
        self.d_bias = activation_dx
        return(self.d_inputs)

    def backward_batch(self, d_val:float) -> tuple[list[float], list[float], float]:
        """ Same as backward, but without resetting the d_inputs, weights, instead adding to them """
        activation_dx = self.activationFunc.backward(self.out)*d_val
        self.d_inputs = [activation_dx*w for w in self.weights]         #< d_inputs are the same because they dont get averaged

        self.d_weights = [activation_dx*i+dw for i, dw in zip(self.inputs, self.d_weights)] #< weights get added and will averaged right before
        self.d_bias = activation_dx+self.d_bias
        return(self.d_inputs)
