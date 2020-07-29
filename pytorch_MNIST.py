#!/usr/bin/env python
# %matplotlab inline
# %config InlineBackend.fig_format = "retina"

import numpy as np
import matplotlab.pyplot as plt
import helper
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from printdescribe import print2, describe2, changepath
from .ml1o1 import sigmoid_activation

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


# transformer to transform and normalize
transformer = transforms.Compose([Transforms.ToTensor(), 
                                 transforms.normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5))])

traindata = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transfomer)
traindownloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)
                                  
# create an iterator to read the dataset                                
iterloader = iter(trainloader)
img, labels = iterloader.next()

print2(type(img), type(labels), img.shape, labels.shape)

# display the image
plt.imshow(img[1].numpy().squeeze(), cmap="blue_r")
plt.show()

# Flatten the 2D images to 1D images
flat1d_img = img.view(img.shape[0], -1)

# create model parameters
input_size = flat1d_img.shape[1]
n_hiddenlayers = 256
n_output = 10

# create weights
feature_weights = torch.randn((input_size, n_hiddenlayers))
hiddenlayer_weights = torch.randn((n_hiddenlayers, n_output))

# create biases
feature_bias = torch.randn((1, n_hiddenlayers))
hiddenlayer_bias = torch.randn((1, n_output))


## y = f2(f1(xW1)W2)

h = sigmoid_activation(torch.mm(flat1d_img,features_weights) + 
                                          features_bias))
y = sigmoid_activation(torch.mm(h, hiddenlayer_bias) + hiddenlayer_bias)

print2(y)

# apply softmax to get the probabilities
prob = softmax_activation(y)

print2(prob.shape, type(prob), prob.sum(dim=1), torch.sum(prob, dim=1))

# define new class
class MyNeuroNetwork(nn.Module):
  
  _inputs = 784
  _neuron = 256
  _output = 10
  
  def __init__(self):
    super().__init__()
    
    # define inputs to hidden layer
    self.hidden = nn.Linear(MyNeuroNetwork._inputs, MyNeuroNetwork._neurons)
    
    # define the output layers
    self.output = nn.Linear(MyNeuroNetwork._neurons, MyNeuroNetwork._output)
    
    # define the sigmoid and softmax functions
    #     self.sigmoid = nn.Sigmoid()
    #     self.softmax = nn.Softmax(dim=1)
    #     using torch.nn.functional, no need to define sigmoid and softmax function
    
  def forward(self, x):
    # using torch.nn.functional
    #     x = self.hidden(x)
    #     x = self.Sigmoid(x)
    #    define hidden layer with sigmoid activation function
    x = F.sigmoid(self.hidden(x))
    
    #     x = self.output(x)
    #     x = self.Softmax(x)
    #     define output layer with softmax activation function
    x = F.softmax(self.output(x), dim=1)
    
    return x
  
model2 = MyNeuroNetwork()
print(model)

# create the feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# define loss function
# criterion = nn.CrossEntropyloss()
criterion = nn.NLLLoss()

# create an iterator to read the dataset  
# img, labels = next(iter(trainloader))
img, labels = next(iterloader)

# Flatten the 2D images to 1D images
flat1d_img = img.view(img.shape[0], -1)

# forward pass, with flattened input data
logprobs = model(flat1D_img)

# compute the loss
loss = criterion(logprobs, labels)
print2(loss)

# compute the gradient descent and backward pass
loss.backward()

# initialize the optimizer
optimin = optim.SGD(model.parameters(), lr=0.01)

#
# Using model2
img, labels = next(iter(iterloader))

img.resize_(img.shape[0], -1)

# clear the gradients
optimin.zero_grad()

# compute the forward pass
output = model2.forward(img)

# compute loss
loss = criterion(output, labels)

# compute backward pass
loss.backward()

# update weights with optimizer
optimin.step()


########################################################################################
########################################################################################
epochs = 6
for epo in range(epochs):
  iterm_loss = 0
  for img, labels in iterloader:
    img.view_(img.shape[0], -1)
    optimin.zero_grad()

    # compute the forward pass
    output = model2.forward(img)

    # compute loss
    loss = criterion(output, labels)

    # compute backward pass
    loss.backward()
    
    iterm_loss += loss.item()
  else:
    print2(f'Training loss: {item_loss/len(iterloader}')
    
