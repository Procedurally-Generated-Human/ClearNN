import numpy as np
from clearNN.layer import Layer

from clearNN.functions import Sigmoid, ReLU, Linear



l1 = Layer(10,2,ReLU)
l2 = Layer(10,10,Sigmoid)
x = np.array([[2, 3]])

a1 = l1.calculate(x)
a2 = l2.calculate(a1)

print(a1)
print("----------")
print(a2)
