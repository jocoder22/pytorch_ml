#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import helper
import torch as tch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from printdescribe import print2, describe2, changepath

plt.style.use("dark_background")

# transformer to transform and normalize
transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# download training dataset
traindata = datasets.FashionMNIST(
    "~/.pytch/F_MNIST_data/",
    download=True,
    train=True,
    transform=transformer,
)
traindownloader = tch.utils.data.DataLoader(
    traindata, batch_size=64, shuffle=True
)


# download test dataset
testdata = datasets.FashionMNIST(
    "~/.pytch/F_MNIST_data/",
    download=True,
    train=False,
    transform=transformer,
)
testdownloader = tch.utils.data.DataLoader(
    testdata, batch_size=64, shuffle=True
)

# view the images
img, label = next(iter(traindownloader))
helper.imshow(img[10,:])
print2(label[10])
plt.show()



# define new classifier class
class MyNeuroNetwork(nn.Module):

    _inputs = 784
    _neuron1 = 128
    _neuron2 = 64
    _neuron3 = 32
    _output = 10

    def __init__(self):
        super().__init__()

        # define inputs to hidden layer
        self.hidden1 = nn.Linear(
            MyNeuroNetwork._inputs, MyNeuroNetwork._neuron1
        )
        self.hidden2 = nn.Linear(
            MyNeuroNetwork._neuron1, MyNeuroNetwork._neuron2
        )
        self.hidden3 = nn.Linear(
            MyNeuroNetwork._neuron2, MyNeuroNetwork._neuron3
        )
        # define the output layers
        self.output = nn.Linear(
            MyNeuroNetwork._neuron3, MyNeuroNetwork._output
        )

    def forward(self, x):
        # flatten the input tensor
        x = x.view(x.shape[0], -1)
        x = tch.relu(self.hidden1(x))
        x = tch.relu(self.hidden2(x))
        x = tch.relu(self.hidden3(x))
        x = tch.log_softmax(self.output(x), dim=1)

        return x

# create device to use cuda
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")

# Initialize the  neural network, define the criterion and optimizer
model = MyNeuroNetwork()
criterion = nn.NLLLoss()
optimin = optim.Adam(model.parameters(), lr=0.008)

# transfer model to device
model.to(device)

def traniner(epoo=5):
    # initialize number of epochs
    epochs = epoo

    for epo in range(epochs):
        # create loss placeholder
        iterm_loss = 0

        # iterate through the data
        for img, labels in traindownloader:
            img, labels = img.to(device), labels.to(device)

            # clear the gradients
            optimin.zero_grad()

            # compute the forward pass
            output = model(img)

            # compute loss
            loss = criterion(output, labels)

            # store the loss
            iterm_loss += loss.item()

            # compute backward pass
            loss.backward()

            # update weights
            optimin.step()

        else:
            print2(
                f"Epoch {epo+1}/{epochs}..>..>.>  " +
                f"Training loss:{iterm_loss/len(traindownloader):.4f}"
            )

# train the neural network
traniner(2)

# use trained model to make prediction
# load next dataset
img, label = next(iter(testdownloader))

# selection image at number 5
img = img[5]

# turn off the gradients, we are just making predictions
with tch.no_grad():
    # predict the image
    logpreds = model(img)

# compute the probabilities, our results are log probabilities
prob = tch.exp_(logpreds)

# display the results
helper.view_classify(img, prob, version='Fashion')
plt.show()