#!/usr/bin/env python3
import gym
import P7_RL_env_v01
import random
import numpy as np
import time
import os

#from stable_baselines.common.policies import MlpPolicy #PPO
from stable_baselines.sac.policies import MlpPolicy # SAC
from stable_baselines import SAC
#from stable_baselines import PPO1

def read_file(filename):
    with open(filename) as fp:
        return int(fp.read())

RLmodel = "SAC"
# Make sure to increment this with each run
simRunID = RLmodel + "_200Ep_OldRewardFunction"

# Creating file structure
simLogPath = "Logging/"+simRunID+"/"
if not os.path.exists(simLogPath): #Create directories
    os.makedirs(simLogPath + 'models/', exist_ok=True)
    os.makedirs(simLogPath + 'log/', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/collision', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/timeout', exist_ok=True)
    os.makedirs(simLogPath + 'screenshots/success', exist_ok=True)
    print('Directory structure created...')
#simRunTime = time.strftime("%Y-%m-%d", time.gmtime()) # File name
simName = '2020.01.28.SAC_200EpisodeDifferentStart_RarePillars'
epFile = simLogPath + 'logs/eps.txt'
currentEp = 0
maxEpisodes = 100000

tensorboardPath = simLogPath + 'sac_P7v1_tensorboard/'

env = gym.make('P7_RL-v0')
print('Environment created...')
if RLmodel == "SAC":
    model = SAC.load(simLogPath + 'models/'+simName, env, tensorboard_log=tensorboardPath)
elif RLmodel == "PPO":
    model = PPO1.load(simLogPath + 'models/'+simName, env, tensorboard_log=tensorboardPath)
print('Model loaded...')
obs = env.reset()
while currentEp<maxEpisodes:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()
  if dones:
    currentEp += 1
    env.reset()
 

 