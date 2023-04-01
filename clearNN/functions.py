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


class Linear(ActivationFunction):
    
    @staticmethod
    def __call__(x):
        return x


class Softmax(ActivationFunction):
    @staticmethod
    def __call__(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()