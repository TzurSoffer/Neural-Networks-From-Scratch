import math
import PYTHON_Network.Mathlib as Mathlib

class Pass:
    """ returns x """
    @staticmethod
    def forward(val) -> float:
        return(val)

    @staticmethod
    def backward(*args) -> float:
        """ derivative of x with respect to x """
        return(1.0)

class ReLU:
    """ Returns 0 if values and x if its not """
    @staticmethod
    def forward(val: float) -> float:
        return(max(0.0, val))    #< by clipping 0, you de-linearize your data
    
    @staticmethod
    def backward(val: float) -> float:
        """ partial derivative of max(x,0) with respect to x """
        return(1.0 if val > 0.0 else 0.0)

class LeakyReLU:
    """ ReLU with small negative slope to prevent dying ReLU problem. Gradient flows even when val < 0 """
    alpha = 0.01  #< small negative slope
    
    @staticmethod
    def forward(val: float) -> float:
        return(val if val > 0.0 else LeakyReLU.alpha * val)
    
    @staticmethod
    def backward(val: float) -> float:
        """ partial derivative of LeakyReLU with respect to x """
        return(1.0 if val > 0.0 else LeakyReLU.alpha)

class Softmax:
    """ converts any output to be squashed from 0 to 1 and also had a nice derivative when paired with  Cross-Entropy """
    @staticmethod
    def forward(vals: list[float]) -> list[float]:
        vals = [math.e**val for val in vals]
        vals = Mathlib.normalize(vals)
        return(vals)

    @staticmethod
    def backward(vals):
        """ derivative of e^x/(sum(e^x) for x in vals) with respect to x
        For proof, see image above or in 'proofs_math/ActivationFuncs'"""
        jacobian = []
        for i, val in enumerate(vals):
            row = []
            for j in range(len(vals)):
                if j == i:
                    row.append(val*(1-val))
                else:
                    row.append(-val*vals[j])
            jacobian.append(row)
        return(jacobian)

class Softmax_batch:
    @staticmethod
    def forward(vals: list[list[float]]) -> list[list[float]]:
        """ same as regular forward, but for a batch """
        return([Softmax.forward(val) for val in vals])

    @staticmethod
    def backward(vals):
        return([Softmax.backward(val) for val in vals])

class ProtectedSoftmax(Softmax):
    """ Softmax but without any overflow from e^x being too large """
    @staticmethod
    def forward(vals):
        """ softmax, but between 0 and 1 """
        maxVal = max(vals)
        return(Softmax.forward([val-maxVal for val in vals]))

class ProtectedSoftmax_batch(Softmax_batch):
    @staticmethod
    def forward(vals):
        return([ProtectedSoftmax.forward(val) for val in vals])
