import random

def zeroes(*shape):
    """ Creates a matrix of zeroes with the given shape """
    if len(shape) == 1:
        return([0.0]*shape[0])
    else:
        return([zeroes(*shape[1:]) for i in range(shape[0])])

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

def addTwoMatrices(m1:list[list[float]], m2:list[list[float]]) -> list[list[float]]:
    out = []
    for r1, r2 in zip(m1, m2):
        out.append(addTwoVectors(r1, r2))
    return(out)

def vectorScalerMult(vector: list[float], scale: float) -> list[float]:
    if type(vector[0]) == list:
        return([vectorScalerMult(v, scale) for v in vector])
    return([v*scale for v in vector])
def scale(vector: list[float], scale: float) -> list[float]:
    return(vectorScalerMult(vector, scale))

def transpose(m) -> list:
    return(list(zip(*m)))
def T(m):
    return(transpose(m))

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

def hilbertIndexToXy(index, order):
    n = 1 << order
    x = y = 0
    t = index
    s = 1

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)

        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        x += s * rx
        y += s * ry

        t //= 4
        s *= 2

    return x, y

def hilbertFlatten(array2d):
    n = len(array2d)
    if any(len(row) != n for row in array2d):            #< Check that the array is square and size is a power of 2
        print(n, len(array2d[0]))
        raise ValueError("Array must be square")

    if n & (n - 1):
        raise ValueError("Size must be a power of 2")

    order = (n.bit_length() - 1)

    result = []
    for i in range(n * n):
        x, y = hilbertIndexToXy(i, order)
        result.append(array2d[y][x])

    return result

def hilbertUnflatten(data):
    n2 = len(data)
    n = int(n2 ** 0.5)

    if n * n != n2 or n & (n - 1):
        raise ValueError("Length must be a square of a power of 2")

    order = n.bit_length() - 1
    array2d = [[None] * n for _ in range(n)]

    for i, value in enumerate(data):
        x, y = hilbertIndexToXy(i, order)
        array2d[y][x] = value

    return array2d