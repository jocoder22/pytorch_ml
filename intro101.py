import numpy as np
import torch
from printdescribe import print2, describe2, changepath


a = torch.tensor([[23,56 ,78],[90, 12,31]])  # a = np.array([23,56 ,78],[90, 12,31]) 

b = torch.rand((3,2))  # b = np.random.randn(3,2)

print2(a.shape)  # print2(a.shape, a.size())

# Matrix multiplication
a = torch.rand(4, 3)
b = torch.rand(3, 4)

c = torch.matmul(a, b)   # c = a.dot(b)

# Element by element multiplication
aa = torch.rand(3, 4)
d = aa * b   # d = np.multiply(a, b)

print2(c, d)

# Zeroes, ones and identity matrix
z = torch.zeros(4, 5)  # z = np.zeros(4, 5)
oo = torch.ones(5, 6)  # oo = np.ones(2, 3)
idtt = torch.eye(4)   # idtt = np.identity(4)

print2(z, oo, idtt)

# torch-numpy interchange
t = torch.tensor([[4,5,7],[2,9,6]])
n = np.array([[2,3,9],[1,4,6]])

torch_from_numpy = torch.from_numpy(n)
numpy_from_torch = t.numpy()

print2(torch_from_numpy, numpy_from_torch)


# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(90., requires_grad=True)
y = torch.tensor(-51., requires_grad=True)
z = torch.tensor(12., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
t = x + y
f = t * z

# Compute the derivatives
f.backward()

# Print the gradients
print2("Gradient of x is: " + str(x.grad))
print2("Gradient of y is: " + str(y.grad))
print2("Gradient of z is: " + str(z.grad))
print2("Gradient of t is: " + str(t.grad))
print2("Gradient of f is: " + str(f.grad))
