import numpy as np
from clearNN.layers import Layer
from clearNN.network import Network
from clearNN.functions import Sigmoid, ReLU, Linear, Softmax


l1 = Layer(10,ReLU)
l2 = Layer(20,Softmax)
x = np.array([[2, 3]])

a1 = l1(x)
a2 = l2(a1)

net = Network([l1, l2])
print(net(x))
