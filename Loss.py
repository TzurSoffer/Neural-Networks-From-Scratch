import math
import Mathlib
import Activation
import copy

""" Calculates the error of the output (mainly Softmax) using -log(x) """

class Entropy:
    @staticmethod
    def _forward_single(val):
        """ return the loss of the function, caped at a minimum of 16.12 and a 0.00000001 for division purposes, note for non-hot vectors, value still need to be multiplied by t """
        return(-math.log(Mathlib.clipAboveZeroBelowOne(val)))  #< math.log is actually ln

    @staticmethod
    def forward(vals:list[float], targets:list[float]) -> float:
        """ Calculate the cross entropy loss of a softmax output """
        out = 0.0
        for v, t in zip(vals, targets):
            out += t*Entropy._forward_single(v)   #< loss times the weight of the loss, more important = bigger loss
        return(out)

    @staticmethod
    def forward_batch(vals:list[float], target:list[float]) -> float:
        """ Calculate the cross entropy loss of a softmax output, but for batches """
        return(Mathlib.mean([Entropy.forward(val, t) for val, t in zip(vals, target)]))

    @staticmethod
    def forwardSparse(vals:list[float], target: int) -> float:
        """ Forward method optimized for a hot target index """
        return(Entropy._forward_single(vals[target]))

    @staticmethod
    def forwardSparse_batch(vals:list[list[float]], targets: list[int]) -> float:
        """ Forward method optimized for a hot target index, but for batches """
        return(Mathlib.mean([Entropy.forwardSparse(val, t) for val, t in zip(vals, targets)]))

    @staticmethod
    def backward(vals: list[float], targets: list[float]) -> list[float]:
        """ Backward function of -t*ln(val) = -t/val """
        return([-t / Mathlib.clipAboveZeroBelowOne(v) for v, t in zip(vals, targets)])

    @staticmethod
    def backwardSparse(vals, targetIndex):
        """ Backward method optimized for a hot target index """
        out = [0.0]*len(vals)  #< initialize zeroes because t would be zero for every value except the hot and the formula is -t/ln(val)
        out[targetIndex] = -1 / Mathlib.clipAboveZeroBelowOne(vals[targetIndex])
        return(out)

class SoftmaxCrossEntropy:
    def __init__(self):
        self.softmaxOutput = []
        self.target = None       #< either an int or list[int] for batch

    def forward(self, outputs: list[float], target: int) -> float:
        self.target = target
        self.softmaxOutput = Activation.Softmax.forward(outputs)
        return(Entropy.forwardSparse(self.softmaxOutput, target))

    def forward_batch(self, outputs: list[list[float]], targets: list[int]) -> float:
        self.target = targets
        self.softmaxOutput = Activation.Softmax.forward_batch(outputs)
        return(Entropy.forwardSparse_batch(self.softmaxOutput, targets))

    def backward(self, dvalue=1) -> list[float]:
        ## derivative of err(softmax) = softmaxVector-targetVector
        self.softmaxOutput[self.target] -= 1
        if dvalue != 1:
            self.softmaxOutput = [val*dvalue for val in self.softmaxOutput]
        return(self.softmaxOutput)

    def backward_batch(self, dvalue=1) -> list[list[float]]:
        batchSize = len(self.softmaxOutput)
        out = []

        for row, t in zip(self.softmaxOutput, self.target):
            copiedRow = copy.copy(row)
            copiedRow[t] -= 1
            copiedRow = [x*dvalue/batchSize for x in copiedRow]
            out.append(copiedRow)

        return(out)
