import math
import Mathlib

""" Calculates the error of the output (mainly Softmax) using -log(x) """

def _forward_single(val):
    """ return the loss of the function, caped at a minimum of 16.12 and a 0.00000001 for division purposes, note for non-hot vectors, value still need to be multiplied by t """
    return(-math.log(Mathlib.clipAboveZeroBelowOne(val)))  #< math.log is actually ln

def forward(vals:list[float], targets:list[float]) -> float:
    """ Calculate the cross entropy loss of a softmax output """
    out = 0.0
    for v, t in zip(vals, targets):
        out += t*_forward_single(v)   #< loss times the weight of the loss, more important = bigger loss
    return(out)

def forward_batch(vals:list[float], target:list[float]) -> list[float]:
    """ Calculate the cross entropy loss of a softmax output, but for batches """
    return([forward(val, t) for val, t in zip(vals, target)])

def forwardSparse(vals:list[float], target: int) -> float:
    """ Forward method optimized for a hot target index """
    return(_forward_single(vals[target]))

def forwardSparse_batch(vals:list[list[float]], targets: list[int]) -> list[float]:
    """ Forward method optimized for a hot target index, but for batches """
    return([forwardSparse(val, t) for val, t in zip(vals, targets)])

def backward(vals: list[float], targets: list[float]) -> list[float]:
    """ Backward function of -t*ln(val) = -t/val """
    return([-t / Mathlib.clipAboveZeroBelowOne(v) for v, t in zip(vals, targets)])

def backward_batch(vals: list[list[float]], targets: list[list[float]]) -> list[list[float]]:
    """ Backward method for batches on a batch """
    out = [backward(val, t) for val, t in zip(vals, targets)]
    return(out)

def backwardSparse(vals, targetIndex):
    """ Backward method optimized for a hot target index """
    out = [0.0]*len(vals)  #< initialize zeroes because t would be zero for every value except the hot and the formula is -t/ln(val)
    out[targetIndex] = -1 / Mathlib.clipAboveZeroBelowOne(vals[targetIndex])
    return(out)

def backwardCrossEntropy(softmaxOutput: list[float], target: int):
    ## derivative of err(softmax) = softmaxVector-targetVector
    softmaxCopy = [softmaxOutput[i] for i in range(len(softmaxOutput))]
    softmaxCopy[target] -= 1
    return(softmaxCopy)

def backwardCrossEntropy_batch(softmaxOutput: list[list[float]], targets: list[int]) -> list[list[float]]:
    """ Backward method optimized for a hot target index, but for batches """
    return([backwardCrossEntropy(val, t) for val, t in zip(softmaxOutput, targets)])
