import numpy as np
from .functions import ActivationFunction

class Layer:

    def __init__(self, size:int, activation_function:ActivationFunction):
        self.size = size
        self.activation_function = activation_function
        #self.w = np.ones([feature_size, size])
        #self.b = np.zeros([1,size])
        self.w = None
        self.b = None

    def __call__(self, input):
        if self.w is None:
            self.w = np.ones([input.shape[1], self.size])
            self.b = np.ones([1, self.size])
        return self.activation_function(np.dot(input, self.w) + self.b)
