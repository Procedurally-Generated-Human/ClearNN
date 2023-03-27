import numpy as np

class Layer:

    def __init__(self, size:int, feature_size:int, activation_function:str):
        self.size = size
        self.feature_size = feature_size
        self.activation_function = activation_function
        self.w = np.ones([feature_size, size])
        self.b = np.zeros([1,size])
    

    def calculate(self, input):
        return np.dot(input, self.w) + self.b
