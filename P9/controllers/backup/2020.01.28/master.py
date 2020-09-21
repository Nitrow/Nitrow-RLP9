#!/home/mark/P7_RL/bin/python
import gym
import P7_RL_env_v01
import random
import numpy as np
import time
import os

#from stable_baselines.common.policies import MlpPolicy #PPO
from stable_baselines.sac.policies import MlpPolicy # SAC
from stable_baselines import SAC
from stable_baselines import PPO1

def read_file(filename):
    with open(filename) as fp:
        return int(fp.read())

# Make sure to increment this with each run
simRunID = "2020-01-24_WeekendTraining"

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
simName = 'P7_NoObstacles'
epFile = simLogPath + 'logs/eps.txt'
currentEp = 0
maxEpisodes = 1
timeStepsPerLoad = 60000
tensorboardPath = simLogPath + 'sac_P7v1_tensorboard/'

env = gym.make('P7_RL-v0')
print('Environment created...')
# Check how many episodes the model has been trained for:
if os.path.exists(epFile):
  currentEp = read_file(epFile)
  print('Resuming training from episode {}'.format(currentEp))

#  Check if training log of previous training exists, load model if does
if currentEp <= maxEpisodes:
  if os.path.exists('logs/' + simName + '.txt'):
    print('Loading previous model...')
    model = SAC.load(simLogPath + simName, env, tensorboard_log=tensorboardPath)
  else:
    print('Creating new model...')
    model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=tensorboardPath)
  model.learn(total_timesteps=timeStepsPerLoad, log_interval=1)
  print('Training finished...')
  model.save(simLogPath +'models/'+ simName)
  print('Model saved...')
  env.SaveAndQuit()

model = SAC.load(simName, env, tensorboard_log=tensorboardPath)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
      env.reset()
    
 

 