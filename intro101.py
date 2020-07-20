import numpy as np
import torch as tch
from printdescribe import print2, describe2, changepath


a = tch.tensor([[23,56 ,78],[90, 12,31]])  # a = np.array([23,56 ,78],[90, 12,31]) 

b = tch.rand(3,2)  # b = np.random.randn(3,2)

print2(a.shape)  # print2(a.shape, a.size())


# Matrix multiplication
c = a.matmul(b)   # c = a.dot(b)

# Element by element multiplication

d = a * b   # d = np.multiply(a, b)


# Zeroes, ones and identity matrix
z = tch.zeros(4, 5)  # z = np.zeros(4, 5)
oo = tch.ones(2, 3)  # oo = np.ones(2, 3)
idtt = tch.eye(4)   # idtt = np.identity(4)

# torch-numpy interchange
t = tch.tensor([4,5,7],[2,9,6])
n = np.array([2,3,9],[1,4,6])

torch_from_numpy = tch.from_numpy(n)
numpy_from_torch = t.numpy()


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
