#!/usr/bin/env python
# %matplotlab inline
# %config InlineBackend.fig_format = "retina"

import numpy as np
import matplotlib.pyplot as plt
import helper
import torch as tch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from printdescribe import print2, describe2, changepath
from pytorchFunctions import sigmoid_activation, softmax_activation

plt.style.use("dark_background")

# transformer to transform and normalize
transformer = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize((0.5,), (0.5,))])

traindata = datasets.MNIST('~/.pytch/MNIST_data/', download=True, train=True, transform=transformer)
traindownloader = tch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)
                                  
# create an iterator to read the dataset                                
iterloader = iter(traindownloader)
img, labels = iterloader.next()

print2(type(img), type(labels), img.shape, labels.shape)

# display the image
plt.imshow(img[1].numpy().squeeze(), cmap='Greys')
plt.show()

# Flatten the 2D images to 1D images
flat1d_img = img.view(img.shape[0], -1)

# create model parameters
input_size = flat1d_img.shape[1]
n_hiddenlayers = 256
n_output = 10

# create weights
feature_weights = tch.randn((input_size, n_hiddenlayers))
hiddenlayer_weights = tch.randn((n_hiddenlayers, n_output))

# create biases
feature_bias = tch.randn((1, n_hiddenlayers))
hiddenlayer_bias = tch.randn((1, n_output))


## y = f2(f1(xW1)W2)

h = sigmoid_activation(tch.mm(flat1d_img,feature_weights) + 
                                          feature_bias)
y = sigmoid_activation(tch.mm(h, hiddenlayer_weights) + hiddenlayer_bias)

print2(y)

# apply softmax to get the probabilities
prob = softmax_activation(y)

print2(prob.shape, type(prob), prob.sum(dim=1), tch.sum(prob, dim=1))

# define new class
class MyNeuroNetwork(nn.Module):
  
  _inputs = 784
  _neuron1 = 128
  _neuron2 = 64
  _neuron3 = 32
  _output = 10
  
  def __init__(self):
    super().__init__()
    
    # define inputs to hidden layer
    self.hidden1 = nn.Linear(MyNeuroNetwork._inputs, MyNeuroNetwork._neuron1)
    self.hidden2 = nn.Linear(MyNeuroNetwork._neuron1, MyNeuroNetwork._neuron2)
    self.hidden3 = nn.Linear(MyNeuroNetwork._neuron2, MyNeuroNetwork._neuron3)
    # define the output layers
    self.output = nn.Linear(MyNeuroNetwork._neuron3, MyNeuroNetwork._output)
    
    # define the sigmoid and softmax functions
    #     self.sigmoid = nn.Sigmoid()
    #     self.softmax = nn.Softmax(dim=1)
    #     using tch.nn.functional, no need to define sigmoid and softmax function
    
  def forward(self, x):
    #  define hidden layer with sigmoid activation function
    #  x = self.hidden(x)
    #  x = self.Sigmoid(x)

    # using tch.nn.functional
    # x = F.sigmoid(self.hidden1(x))
    # x = F.sigmoid(self.hidden2(x))
    # x = F.sigmoid(self.hidden3(x))


    # "nn.functional.sigmoid is deprecated. Use torch.sigmoid instead."
    x = tch.sigmoid(self.hidden1(x))
    x = tch.sigmoid(self.hidden2(x))
    x = tch.sigmoid(self.hidden3(x))


    # define output layer with softmax activation function
    #  x = self.output(x)
    # x = self.Softmax(x)
    # x = F.softmax(self.output(x), dim=1)
    x = tch.softmax(self.output(x), dim=1)
    
    return x
  
model2 = MyNeuroNetwork()
print2(model2)

# the weights and biases are added automatically
print2(model2.hidden1.weight, model2.hidden1.bias)

# set the weights and biases
model2.hidden1.bias.data.fill_(0)
model2.hidden1.weight.data.normal_(std=0.01)
print2(model2.hidden1.weight, model2.hidden1.bias)



########################################################################
# Using model
img, labels = next(iter(iterloader))

img = img.view(img.shape[0], 1, -1)

# compute the forward pass
img0 = 0
output = model2.forward(img[img0,:])

img = img[img0]
helper.view_classify(img.view(1, 28, 28), output)
plt.show()


########################################################################
# Using sequential model
# create the feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 86),
                      nn.ReLU(),
                      nn.Linear(86, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.Softmax(dim=1))
print2(model)
print2(model[0], model[0].weight)

# get some images and labels
img, labels = next(iter(iterloader))

img.resize_(img.shape[0], 1, 784)

# compute the forward pass
img0 = 5
output = model.forward(img[img0,:])

helper.view_classify(img[img0].view(1, 28, 28), output)
plt.show()   



########################################################################################
########################################################################################
########################################################################################

# define loss function
# criterion = nn.CrossEntropyloss()
criterion = nn.NLLLoss()

# create an iterator to read the dataset  
# img, labels = next(iter(trainloader))
img, labels = next(iterloader)

# Flatten the 2D images to 1D images
flat1d_img = img.view(img.shape[0], -1)

# forward pass, with flattened input data
logprobs = model(flat1d_img)

# compute the loss
loss = criterion(logprobs, labels)
print2(loss)

# compute the gradient descent and backward pass
loss.backward()

# initialize the optimizer
optimin = optim.SGD(model.parameters(), lr=0.02)

#
# Using model2
img, labels = next(iter(iterloader))

img.resize_(img.shape[0], -1)

# clear the gradients
optimin.zero_grad()

# compute the forward pass
output = model.forward(img)

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
    
    # clear the gradients
    optimin.zero_grad()

    # compute the forward pass
    output = model.forward(img)

    # compute loss
    loss = criterion(output, labels)

    # compute backward pass
    loss.backward()
    
    # update weights
    optimin.step()
    
    iterm_loss += loss.item()
    
  else:
    print2(f'Training loss: {item_loss/len(iterloader)}')
    
# view the prediction
img, label = next(item(iterloader))

img = img[0].view(1, img.shape[0])

with tch.no_grad():
  loggit = model.forward(img)
  
prob = F.softmax(loggit, dim=1)
helper.view_classify(img.view(1, 28, 28), prob)                         
                      
