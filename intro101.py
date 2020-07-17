import numpy as np
import torch as tch
from printdescribe import print2, describe2, changepath


a = tch.tensor([23,56 ,78],[90, 12,31])  # a = np.array([23,56 ,78],[90, 12,31]) 

b = tch.rand(3,2)  # b = np.random.randn(3,2)

print2(a.shape, tch.Size(a))  # print2(a.shape, a.size())


# Matrix multiplication
c = a.matmul(b)   # c = a.dot(b)

# Element by element multiplication

d = a * b   # d = np.multiply(a, b)


# Zeroes, ones and identity matrix
z = tch.zeros(4, 5)  # z = np.zeros(4, 5)
oo = tch.ones(2, 3)  # oo = np.ones(2, 3)
idtt = tch.eye(4)   # idtt = np.identity(4)
