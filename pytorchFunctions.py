#!/usr/bin/env python
import torch

def sigmoid_activation(x):
  """ The Sigmoid activation function that result probabilities
    
    Input:
      x: torch.Tensor
      
    Output:
      y: float
  
  """
  
  y = 1/(1 + torch.exp(-x))
  
  return y


def softmax_activation(x):
  """ The softmax_activation function
  
    Inputs:
      X: torch.Tensor
      
    Output:
      p: torch.Tensor (probabilities)
      
    Numpy Implementation:
      deno = np.exp(x).sum(axis=1).reshape(-1, 1)
      return np.divide(np.exp(x), deno)
  
  """
  p = torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
  
  return p
