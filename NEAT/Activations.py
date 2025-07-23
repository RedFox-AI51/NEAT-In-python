# Activations.py
import math
from random import choices

class ActivationFunctions:
    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + math.exp(-x))


    @staticmethod
    def ReLu(x):
        return max(0, x)

    @staticmethod
    def TanH(x):
        return math.tanh(x)

    @staticmethod
    def Sine(x):
        return math.sin(x)

    @staticmethod
    def Linear(x):
        return x

    @staticmethod
    def Gaussian(x):
        return math.exp(-x**2)

    @staticmethod
    def random_activation():
        activations = [
            ActivationFunctions.Sigmoid,
            ActivationFunctions.ReLu,
            ActivationFunctions.TanH,
            ActivationFunctions.Sine,
            ActivationFunctions.Linear,
            ActivationFunctions.Gaussian
        ]
        return choices(activations, weights=[0.8, 0.8, 0.8, 0.5, 0.4, 0.4])[0]