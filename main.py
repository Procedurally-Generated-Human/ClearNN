import numpy as np
from clearNN.layers import Layer
from clearNN.network import Network
from clearNN.activation_funcs import Sigmoid, ReLU, Linear, Softmax
from clearNN.cost_funcs import MeanSquareError


l1 = Layer(3,ReLU)
l2 = Layer(3,ReLU)
x = np.array([[2, 3],[2,2]])

a1 = l1(x)
a2 = l2(a1)

net = Network([l1, l2])
print(net(x))

print(MeanSquareError(net(x), np.array([[20,20,20],[20,20,20]])))