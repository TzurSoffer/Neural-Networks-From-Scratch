import math

def RecLE(val):
    return(max(0, val))

def none(val):
    return(val)

def softmax(val):
    return(math.e**val)

def protectedSoftmax(vals):
    out = []
    maxVal = max(vals)
    for v in vals:
        out.append(softmax(v-maxVal))
    return(out)
