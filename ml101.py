#!/usr/bin/env python
import torch
from printdescribe import print2, describe2, changepath
from pytorchFunctions import sigmoid_activation

# set the seed
torch.manual_seed(90)

# generate features vector
features = torch.randn((1,10))

# generate weights
weights = torch.randn_like(features)

# generate bias
bias = torch.randn((1,1))

# computer the prediction
prob = sigmoid_activation(torch.sum(features * weights) + bias)
# also can do sigmoid_activation((features * weights).sum() + bias)

prob2 = sigmoid_activation(torch.mm(features, weights.view(-1,1)) + bias)

if __name__ == "__main__":
  print2(prob, prob2)


## y = f2(f1(xW1)W2)

