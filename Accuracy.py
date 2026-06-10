import math
try:
    import mathlib as Mathlib   #< c++ version (much faster training). Must be compiled first
except ImportError:
    import Mathlib

class Accuracy_hard:
    """ calculates how accurate the the result is, for the hard version, it gets an array of index (must be all 0 and a single value has to be 1), and it returns 1 if that most likely output is at the correct index and 0 if its not """
    def __init__(self):
        self._length = 0
        self.correct = 0
        self.accuracy = 0.0

    @staticmethod
    def calc(result, target):
        if Mathlib.argmax(result) == target:
            return(1)
        return(0)

    def epoch(self, result, target):
        if type(target) == int:    #< convert to batch mode, I am not using a separate function because its not critical to efficiency
            target = [target]
            result = [result]
        for r, t in zip(result, target):
            self.correct += Accuracy_hard.calc(r, t)
            self._length += 1
            self.accuracy = self.correct/self._length
        return(self.accuracy)

class Accuracy_soft:
    """ calculates how accurate the the result is, for the soft version, it get an array of target results, ex: [0.2, 0.8, 0.0, 0.0] and returns a number to quantify how close it is"""
    def __init__(self):
        self._length = 0
        self.correct = 0.0
        self.accuracy = 0.0

    @staticmethod
    def calc(result, target):
        dot = Mathlib.dot2Vectors(result, target)
        norm_r = math.sqrt(sum(r*r for r in result))
        norm_t = math.sqrt(sum(t*t for t in target))
        soft = dot / (norm_r * norm_t)      #< normalize between 0 and 1
        return(soft)

    def epoch(self, result, target):
        soft = Accuracy_soft.calc(result, target)
        self._length += 1
        self.correct += soft
        self.accuracy = self.correct/self._length
        return(self.accuracy)
        