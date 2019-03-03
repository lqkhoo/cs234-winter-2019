import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """
    Flatten any tensor into a vector.
    """
    def forward(self, x):
        return x.view(x.size()[0], -1)


class PolicyNet(object):
    """
    Feedfoward net containing policy. Architecture from Mordatch NIPS 2015
    """
    def __init__(self, input_shape, output_shape):
        self.fc1 = nn.Linear(1, 250) # todo change input dimension
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.out = nn.Linear(250, 1) # todo change output dimension

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)
        x = nn.ReLU(x)
        x = self.fc3(x)
        x = nn.ReLU(x)
        x = self.out(x)
        return x


class MemoryNet(object):
    """
    Recurrent net containing memory state. Architecture not stated.
    """

    def __init__(self):
        pass # todo
