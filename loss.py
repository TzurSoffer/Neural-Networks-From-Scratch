import math
import mathlib

def calcLoss(vals, index):
    return(-math.log(mathlib.clip(vals[index], 1e-7, 1-1e-7)))