import torch as tch
from printdescribe import print2, describe2, changepath


a = tch.tensor([23,56 ,78],[90, 12,31])

b = tch.rand(3,2)

print2(a.shape, tch.Size(a))
