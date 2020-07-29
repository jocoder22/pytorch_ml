#!/usr/bin/env python
# %matplotlab inline
# %config InlineBackend.fig_format = "retina"

import numpy as np
import matplotlab.pyplot as plt
import helper
import torch
from torchvision import datasets, transforms

from printdescribe import print2, describe2, changepath
from .ml1o1 import sigmoid_activation


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
                                  
                
                                   
                           
