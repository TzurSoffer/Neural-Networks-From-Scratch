import math

def calcLoss(vals):
    return([-math.log(v) for v in vals])