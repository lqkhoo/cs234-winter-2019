from mujoco_py import load_model_from_path, MjSim, functions
import mujoco_py.generated.const as const
import math
import os
import numpy as np
from src.indexer import Index
from src.viewer import Viewer
from src.nets import Flatten
from src.task import Task

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



class PointMass(Task):
    """
    Given a point mass and a stationary target both constrained
    on the x-y plane, move the point mass to the target and 
    have it stop there in the shortest possible time.

    The optimal controller for this task is:
    let a_max = maximal acceleration

    v^2 = u^2 - 2as. Since v=0, s = u^2 / 2a
    Shortest possible time is to accelerate maximally
    until distance to target == minimum stopping 
    distance with maximal deceleration. From then on,
    decelerate maximally until rest.

    The optimal controller is:
    a = a_max: if d > u^2 / 2a_max
    a = -a_max: otherwise
    """

    def __init__(self, mjsim, render=False):
        super().__init__(mjsim, render)

        # Convenience accessors
        self.ctrl = self.data.ctrl
        self.time_resolution = self.model.opt.timestep
        self.time = self.data.time

        self.ball = self.idx.id_of('ball', const.OBJ_BODY)
        self.target = self.idx.id_of('target', const.OBJ_BODY)
        self.actuator_x = self.idx.id_of('actuator-x', const.OBJ_ACTUATOR)
        self.actuator_y = self.idx.id_of('actuator-y', const.OBJ_ACTUATOR)

        self.ball_pos = self.data.body_xpos[self.ball]
        self.ball_vel = self.data.cvel[self.ball]
        self.target_pos = self.data.body_xpos[self.target]

        assert(self.model.actuator_ctrllimited[self.actuator_x] == 1)
        assert(self.model.actuator_ctrllimited[self.actuator_y] == 1)
        # Todo: find max acceleration
        # Todo: find 'mass'
        self.accel = self.data.qacc_unc
        
        self.displacement = self.ball_pos - self.target_pos


    def reset(self):
        self.sim.reset()



        # Initialize the target to within some distance d1 of the origin
        D1 = 0 # Mean distance between target and origin (per dimension)
        S1 = 5 # Sdev of distance between target and origin (per dimension)
        self.target_pos = D1 + np.random.normal(0, 1, (2, )) * S1
        # the ball to within some distance d2 of the target
        D2 = 5
        S2 = 10
        self.ball_pos = D2 + np.random.normal(0, 1, (2, )) * S2
        # and finally the ball to some random gaussian initial velocity
        V3 = 0
        S3 = 5
        self.ball_vel = V3 + np.random.normal(0, 1, (2, )) * S3


    
    def expert_output(self):

        m = 1.19366207
        a_max = 1.0 / m

        for dim in range(3):
            ball_pos = self.data.body_xpos[self.ball]
            target_pos = self.data.body_xpos[self.target]
            displacement = target_pos - ball_pos
            ball_vel = self.data.qvel[0:2]
            sign = 1.0 if displacement[dim] > 0 else -1

            if dim >= 2: # ignore z-axis
                continue
            stopping_distance = ball_vel[dim] * ball_vel[dim] / (2 * a_max)
            
            if np.abs(displacement[dim]) > stopping_distance:
                self.ctrl[dim] = sign * a_max
            else:
                self.ctrl[dim] = -ball_vel[dim] * ball_vel[dim] / 2 / 1.19366207
    


    def reward(self, timestep):
        pass

    
    def run(self):

        while True:

            if self.render:
                self.viewer.render()
            if self.time >= 100:
                break

            self.expert_output()

            ball_pos = self.data.body_xpos[self.ball]
            target_pos = self.data.body_xpos[self.target]
            displacement = target_pos - ball_pos
            ball_vel = self.data.qvel
            ball_accel = self.data.qacc[0:2]
            stopping_distance = ball_vel * ball_vel / 2
            print("a{}\n v{}\n d{}\nY{}".format(ball_accel, ball_vel, displacement, stopping_distance))

            sim.step()



if __name__ == '__main__':
    MODEL_PATH = '../assets/pointmass.xml'
    model = load_model_from_path(MODEL_PATH)
    sim = MjSim(model)
    task = PointMass(sim, render=True)

    task.run()

    """
    input_size = np.prod(x.shape)
    output_size = np.prod(y.shape)
    policy = Policy(input_size, output_size)
    """

    buffer = []
    buffer_max_size = 100000
    buffer_idx = 0
