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

print2(tch.cuda.is_available())


# transformer to transform and normalize
transformer = transforms.Compose([transforms.ToTensor(), 
                                 transforms.Normalize((0.5,), (0.5,))])

traindata = datasets.MNIST('~/.pytch/MNIST_data/', download=True, train=True, transform=transformer)
traindownloader = tch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)
  
# create an iterator to read the dataset                                
iterloader = iter(traindownloader)
img, labels = iterloader.next()

print2(type(img), type(labels), img.shape, labels.shape)

# Using sequential model
# create the feed-forward network
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")

model = nn.Sequential(nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 86),
                    nn.ReLU(),
                    nn.Linear(86, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    nn.LogSoftmax(dim=1))

# define loss function
# criterion = nn.CrossEntropyloss()
criterion = nn.NLLLoss()

# # initialize the optimizer
optimin = optim.SGD(model.parameters(), lr=0.02)

model.to(device)

# initialize number of epochs
epochs = 3

for epo in range(epochs):
    # create loss placeholder
    iterm_loss = 0

    # iterate through the data
    for img, labels in traindownloader:
        # flatten the images
        # img.resize_(img.shape[0], 1, 784)
         # Move input and label tensors to the default device
        img, labels = img.to(device), labels.to(device)
        img = img.view(img.shape[0], -1)
    
        # clear the gradients
        optimin.zero_grad()

        # compute the forward pass
        output = model(img)

        # compute loss
        loss = criterion(output, labels)

        # compute backward pass
        loss.backward()
        
        # update weights
        optimin.step()
        
        iterm_loss += loss.item()
    
    else:
        print2(f'Epoch {epo + 1}/{epochs}, Training loss: {iterm_loss/len(traindownloader):.4f}')


# use trained model to make prediction
# load next dataset
img, label = next(iter(traindownloader))

# selection image at number 5 and flatten it
img = img[5].view(-1, 784)

# turn off the gradients, we are just making predictions
with tch.no_grad():
    # predict the image
    logpreds = model(img)

# compute the probabilities, our results are log probabilities
prob = tch.exp_(logpreds)

helper.view_classify(img.view(1, 28, 28), prob)
plt.show()   


