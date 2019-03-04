from mujoco_py import load_model_from_path, MjSim, functions
import mujoco_py.generated.const as const
import math
import os
import numpy as np
from src.index import Index
from src.viewer import Viewer
from src.nets import Flatten

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, input_size, output_size, noise_regularization=True):
        super().__init__()

        self.hsize = 50 # hidden fc layer size
        self.noise_mean = 0
        self.noise_sdev = 1

        self.flatten = Flatten()
        self.fc1 = nn.Linear(input_size, self.hsize)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hsize, self.hsize)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hsize, self.hsize)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(self.hsize, output_size)

    def forward(self, x):
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        if self.noise_regularization:
            x = x * self.noise_sdev * np.random.normal(0, 1, size=np.shape(x)) + self.noise_mean
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        if self.noise_regularization:
            x = x * self.noise_sdev * np.random.normal(0, 1, size=np.shape(x)) + self.noise_mean
        x = self.fc3.forward(x)
        x = self.relu3.forward(x)
        if self.noise_regularization:
            x = x * self.noise_sdev * np.random.normal(0, 1, size=np.shape(x)) + self.noise_mean
        x = self.out.forward(x)
        return x



if __name__ == '__main__':
    MODEL_PATH = '../assets/pointroller.xml'
    model = load_model_from_path(MODEL_PATH)
    idx = Index(model)
    sim = MjSim(model)
    data = sim.data
    viewer = Viewer(sim)
    timestep = model.opt.timestep

    x = np.hstack([data.qpos, data.qvel]) # We use position and velocity as features
    y = data.ctrl
    input_size = np.prod(x.shape)
    output_size = np.prod(y.shape)
    policy = Policy(input_size, output_size)

    buffer = []
    buffer_max_size = 100000
    buffer_idx = 0

    t = 0
    while True:

        x = np.hstack([data.qpos, data.qvel]) # We use position and velocity as features
        y = data.ctrl

        viewer.render()

        t += 1 * timestep



        sim.step()
