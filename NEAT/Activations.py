# Activations.py
import math

class ActivationFunctions:
    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def ReLu(x):
        return max(0, x)

    @staticmethod
    def TanH(x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def Sine(x):
        return math.sin(x)

    @staticmethod
    def Linear(x):
        return x

    @staticmethod
    def Gaussian(x):
        return math.exp(-x**2)
