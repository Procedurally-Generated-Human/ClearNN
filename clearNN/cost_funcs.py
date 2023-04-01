import numpy as np


class CostFunction:

    def __new__(self, x, y):
        return self.__call__(x, y)
    
    @staticmethod
    def __call__(x, y):
        pass


class MeanSquareError(CostFunction):

    @staticmethod
    def __call__(x, y):
        return (1/x.shape[0]) * np.square(x-y).sum()
        