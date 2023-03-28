import numpy as np
from clearNN.layer import Layer

from clearNN.functions import Sigmoid, ReLU, Linear



l1 = Layer(10,ReLU)
l2 = Layer(10,Sigmoid)
x = np.array([[2, 3]])

a1 = l1(x)
a2 = l2(a1)

print(a1)
print("----------")
print(a2)
