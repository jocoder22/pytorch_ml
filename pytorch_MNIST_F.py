#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import helper
import torch as tch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from printdescribe import print2, describe2, changepath