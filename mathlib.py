import random

def randn(*shape):
    pass

def dot2Vectors(vector1, vector2):
    out = 0
    for v1, v2 in zip(vector1, vector2):
        out += v1*v2
    return(out)

def normalize(values):
    out = []
    total = sum(values)
    for v in values:
        out.append(v/total)
    return(out)