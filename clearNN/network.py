from collections.abc import Iterable
from numpy import ndarray
from .layers import Layer


class Network:

    def __init__(self, layers:Iterable[Layer]):
        self.layers = layers
    

    def __call__(self, x:ndarray):
        for layer in self.layers:
            x = layer(x)
        return x

    