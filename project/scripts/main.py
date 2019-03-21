import os
import argparse
import sys
import logging
import time
import numpy as np
import tensorflow as tf
import gym
import scipy.signal
import os
import time
import inspect
from utils.general import get_logger, Progbar, export_plot
from config import get_config
from env import AntEnvX, AntEnvXNeg, AntEnvY, AntEnvYNeg
from pprint import pprint

# Cartpole: obs=4, act=2
# Ant: obs=111, act=8

observation_dim = 4
action_dim = 2

weights_path = 'results/CartPole-v0-baseline/model.weights/'

def build_policy_model(mlp_input, scope):
  with tf.variable_scope(scope):
    n_layers = 1
    size = 16
    output_size = action_dim

    x = mlp_input
    for i in range(n_layers):
      x = tf.layers.dense(x, size, activation=tf.nn.relu, name="hdense"+str(i))
    x = tf.layers.dense(x, output_size, activation=None, name="outdense")
    return x

def run():

    observation_placeholder = tf.placeholder(tf.float32, shape=(None, observation_dim))
    action_placeholder = tf.placeholder(tf.float32, shape=(None, action_dim))

    scope = "CartPole-v0_policy_network"

    x_policy = build_policy_model(observation_placeholder, scope=scope)

    # policy_model = build_policy_model(observation_placeholder)
    saver = tf.train.Saver()

    _variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    pprint(_variables)

    with tf.Session() as sess:
        saver.restore(sess, weights_path)



if __name__ == '__main__':
    run()