import random

def clip(val:float, minVal:float=0.0, maxVal:float=1.0) -> float:
    return(min(maxVal, max(minVal, val)))

def clipAboveZeroBelowOne(val: float) -> float:
    """ clip between 1e-7 and 1-1e-7 """
    return(clip(val, 1e-7, 1-1e-7))

def mean(vals:list[float]) -> float:
    return(sum(vals)/len(vals))

def argmax(vals:list[float]) -> int:
    maxVal = vals[0]
    maxIndex = 0
    for i, v in enumerate(vals):
        if v > maxVal:
            maxVal = v
            maxIndex = i
    return(maxIndex)

def argmin(vals:list[float]) -> int:
    minVal = vals[0]
    minIndex = 0
    for i, v in enumerate(vals):
        if v < minVal:
            minVal = v
            minIndex = i
    return(minIndex)

def dot2Vectors(vector1:list[float], vector2:list[float]) -> float:
    out = 0
    for v1, v2 in zip(vector1, vector2):
        out += v1*v2
    return(out)

def dotVectorMatrix(vector:list[float], matrix:list[list[float]]) -> list[float]:
    out = []
    for row in matrix:
        out.append(dot2Vectors(vector, row))
    return(out)

def addTwoVectors(v1:list[float], v2:list[float]) -> list[float]:
    return([i1+i2 for i1,i2 in zip(v1,v2)])

def vectorScalerMult(vector: list[float], scale: float) -> list[float]:
    return([v*scale for v in vector])

def transpose(m) -> list:
    return(list(zip(*m)))

def elementWiseMult(v1:list[float], v2:list[float]) -> list[float]:
    result = []
    for i1, i2 in zip(v1, v2):
        result.append(i1*i2)
    return(result)

def sumNVectors(vectors):
    vector = []
    for T in zip(*vectors):
        vector.append(sum(T))
    return(vector)

def normalize(values:list[float]) -> list[float]:
    out = []
    total = sum(values)
    for v in values:
        out.append(v/total)
    return(out)

def randomNumber(minimum:float=-10.0, maximum:float=10.0, decimals:int=2) -> float:
    return(round(random.uniform(minimum, maximum), decimals))