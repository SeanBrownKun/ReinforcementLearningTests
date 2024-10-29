import numpy as np
from random import random


a = np.array([0, 0, 0, 1, 0, 0, 0])
b = np.array([1, 2, 3, 0, 0])

c = np.convolve(b, a, mode="same")
print(c)

a = np.array([(random())-(random()) for i in range(10)])
print(a.mean())
