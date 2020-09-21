#!/usr/bin/env python3
import gym
import P9_RL_env_v01
import numpy as np
import os
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy #PPO
#from stable_baselines.sac.policies import MlpPolicy # SAC
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128])


from stable_baselines import SAC
from stable_baselines import PPO1

# Make sure to increment this with each run
simRunID = "PPO_256_256"

# Creating file structure
simLogPath = "Logging/"+simRunID+"/"
if not os.path.exists(simLogPath): #Create directories
    os.makedirs(simLogPath + 'models/', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/collision', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/timeout', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/success', exist_ok=True)
    print('Directory structure created...')

epFile = simLogPath + 'log/eps.txt'
timeStepsToTrain = 1000000
tensorboardPath = simLogPath + 'P7v1_tensorboard/'

env = gym.make('P9_RL-v0')
print('Environment created...')

model = PPO1("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.99, tensorboard_log=tensorboardPath)
#model = PPO1.load(simLogPath +'models/'+ simName, env, tensorboard_log=tensorboardPath) 
model.learn(total_timesteps=timeStepsToTrain, log_interval=1)
print('Training finished...')
model.save(simLogPath +'models/'+ simRunID)
print('Model saved...')
env.supervisor.simulationSetMode(0)
