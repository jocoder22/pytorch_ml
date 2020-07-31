#!/usr/bin/env python
import torch
from printdescribe import print2, describe2, changepath
from pytorchFunctions import sigmoid_activation


# set the seed
torch.manual_seed(90)

# create features
features = torch.randn((1, 10))

# define sizes of layers
input_size = features.shape[1]
n_hiddenlayers = 4
n_output = 1

# create weights
feature_weights = torch.randn((input_size, n_hiddenlayers))
hiddenlayer_weights = torch.randn((n_hiddenlayers, n_output))

# create biases
feature_bias = torch.randn((1, n_hiddenlayers))
hiddenlayer_bias = torch.randn((1, n_output))


# y = f2(f1(xW1)W2)
hiddenlayer_output = sigmoid_activation(
    torch.mm(features, feature_weights) + feature_bias)
y = sigmoid_activation(
    torch.mm(
        hiddenlayer_output,
        hiddenlayer_weights) +
    hiddenlayer_bias)

print2(y)
