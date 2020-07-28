#!/usr/bin/env python
import torch
from printdescribe import print2, describe2, changepath


# set the seed
torch.manual_seed(90)

def sigmoid_activation(x):
  """ The Sigmoid activation function that result probabilities
    
    Input:
      x: torch.Tensor
      
     Output:
       y: float
  
  """
  
  y = 1/(1 + torch.exp(-x))
  
  return y
