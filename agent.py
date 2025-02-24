import tensorflow as tf
import numpy as np
import gym
import math
import os

import model
import architecture as policies
import mario_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


def main():
    config = tf.ConfigProto()

    # Allowing GPU memory growth
    config.gpu_options.allow_growth = True

    with tf.Session(config=config):
        model.learn(policy=policies.A2CPolicy,
                            env=DummyVecEnv([env.make_train_0,env.make_train_1,env.make_train_2,env.make_train_3]), 
                            nsteps=2048, # Steps per environment
                            total_timesteps=10000000,
                            gamma=0.99,
                            lam = 0.95,
                            vf_coef=0.5,
                            ent_coef=0.01, 
                            lr = 2e-4,
                            max_grad_norm = 0.5, 
                            log_interval = 10
                            )

if __name__ == '__main__':
    main()