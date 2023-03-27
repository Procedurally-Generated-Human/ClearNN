import numpy as np
import math


class ActivationFunction:

    def __new__(self, x):
        return self.__call__(x)
    @staticmethod
    def __call__(x):
        pass


class ReLU(ActivationFunction):

    @staticmethod
    def __call__(x):
        return np.maximum(0,x)


class Sigmoid(ActivationFunction):

    @staticmethod
    def __call__(x):
        return 1/(1+ np.exp(x))