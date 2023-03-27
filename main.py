import numpy as np
from clearNN.layer import Layer

from clearNN.functions import functions



l1 = Layer(5,2,"hi")
x = np.array([[2, 3]])
print(functions.Sigmoid(np.array([2,4,0])))
#print(l1.calculate(x))
