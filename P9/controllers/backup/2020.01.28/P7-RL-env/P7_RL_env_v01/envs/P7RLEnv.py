import gym
from gym import spaces
import numpy as np
from controller import Robot, Supervisor, Lidar, Motor
import random
import time
import math
from math import pi
import os

class P7RLEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    motors = []
    timeStep = 32
    restrictedGoals = True
    boxEnv = False
    isFixedGoal = False
    fixedGoal = [3.5,0]
    logging = True
    maxTimestep = 20000
    robotResetLocation = [0, 0, 0]

    def __init__(self):
        # Initialize TurtleBot specifics
        self.maxAngSpeed = 1   # 2.84 max
        self.maxLinSpeed = 0.14 # 0.22 max
        # Initialize reward parameters:
        self.moveTowardsParam = 1
        self.moveAwayParam = 1.1
        self.safetyLimit = 0.5
        self.obsProximityParam = 0.01
        self.faceObstacleParam = 10
        # Total reward counter for each component
        self.moveTowardGoalTotalReward = 0
        self.obsProximityTotalPunish = 0
        self.faceObstacleTotalPunish = 0
        # Initialize RL specific variables
        self.rewardInterval = 1  # How often do you want to get rewards
        self.epConc = ''  # Conculsion of episode [Collision, Success, TimeOut]
        self.reward = 0  # Reward gained in the timestep
        self.totalreward = 0  # Reward gained throughout the episode
        self.counter = 0  # Timestep counter
        self.needReset = True  #
        self.state = []  # State of the environment
        self.done = False  # If the episode is done
        self.action = [0, 0]  # Current action chosen
        self.prevAction = [0,0]  # Past action chosen
        self.goal = []  # Goal
        self.goals = [x/100 for x in range(-450,451,150)]
        self.dist = 0  # Distance to the goal
        self.prevDist = 0  # Previous distance to the goal
        # Logging variables
        self.logDir = ""
        self.path = ""
        self.currentEpisode = 0  # Current episode number
        self.start = 0  # Timer for starting
        self.duration = 0  # Episode duration in seconds
        self.epInfo = []  # Logging info for the episode
        self.startPosition = []
        self.prevMax = 0 # temporary variable
        # Initialize a supervisor
        self.supervisor = Supervisor()
        # Initialize robot
        self.robot = self.supervisor.getFromDef('TB')
        self.translationField = self.robot.getField('translation')
        self.orientationField = self.robot.getField('rotation')
        # Initialize lidar
        self.lidar = self.supervisor.getLidar('TurtleBotLidar')
        self.lidar.enable(self.timeStep)
        self.lidarRanges = []
        # Initialize Motors
        self.motors.append(self.supervisor.getMotor("left wheel motor"))
        self.motors.append(self.supervisor.getMotor("right wheel motor"))
        self.motors[0].setPosition(float("inf"))
        self.motors[1].setPosition(float("inf"))
        self.motors[0].setVelocity(0.0)
        self.motors[1].setVelocity(0.0)
        self.direction = []
        self.position = []
        self.orientation = []
        # Initialize Goal Object
        self.goalObject = self.supervisor.getFromDef('GOAL')
        self.goalTranslationField = self.goalObject.getField('translation')
        # Initialize action-space and observation-space
        self.action_space = spaces.Box(low=np.array([-self.maxLinSpeed, -self.maxAngSpeed]),
                                       high=np.array([self.maxLinSpeed, self.maxAngSpeed]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(24,), dtype=np.float16)

    def reset(self):
        if not self.logDir: self._getLogDirectory()
        self._startEpisode() # Reset evetything needs resetting
        self.supervisor.step(1) # Make it Happen
        self.goal = self._setGoal() # Set Goal
        self._resetObject(self.goalObject, [self.goal[0], 0.01, self.goal[1]])
        self._getState()
        self.startPosition = self.position[:]
        self.state = [self.prevAction[0], self.prevAction[1], self.dist, self.direction] + self.lidarRanges # Set State
        return np.asarray(self.state)

    def step(self, action):
        self.prevAction = self.action[:] # Set previous action     
        self.action = action # Take action
        self._take_action(action)
        self.supervisor.step(1)
        self.counter += 1
        #   Get State(dist from obstacle + lidar) and convert to numpyarray
        self._getState() # Observe new state
        self.state = np.asarray([self.prevAction[0], self.prevAction[1], self.dist, self.direction] + self.lidarRanges)
        self.reward = self._calculateReward() #   get Reward
        self.done, extraReward = self._isDone() #   determine if state is Done, extra reward/punishment
        self.reward += extraReward
        self.totalreward += self.reward
        return [self.state, self.reward, self.done, {}]

    def _trimLidarReadings(self, lidar):
        lidarReadings = []
        new_lidars = [x if x != 0 else 3.5 for x in lidar] # Replace 0's with 3.5 (max reading)
        for x in range(0, 360, 18):
            end = x + 18
            lidarReadings.append(min(new_lidars[x:end])) # Take the minimum of lidar Readings
        return lidarReadings

    def _resetObject(self, object, coordinates):
        object.getField('translation').setSFVec3f(coordinates)
        object.getField('rotation').setSFRotation([0,1,0,0])
        object.resetPhysics()
        
    def _setGoal(self):
        if not self.isFixedGoal:
          while True:
            if self.restrictedGoals:
              gs = random.sample(self.goals,1)
              gs.append(random.randrange(-45, 45)/10)
              g = gs[:] if np.random.randint(2) == 0 else [gs[1], gs[0]]
            elif self.boxEnv:
              xGoal = random.randrange(-40, -25)/10 if np.random.randint(2) == 0 else random.randrange(25, 40)/10
              yGoal = random.randrange(-40, 40)/10
              g = [yGoal, xGoal] if np.random.randint(2) == 0 else [xGoal, yGoal]
            else: 
              xGoal = random.randrange(-40, 40)/10
              yGoal = random.randrange(-40, 40)/10
              g = [xGoal, yGoal]
            pos1 = self.robot.getPosition()[0]
            pos2 = self.robot.getPosition()[2]
            distFromGoal = ((g[0]-pos1)**2 + (g[1]-pos2)**2)**0.5
            if distFromGoal >= 1:
              break
        else:
          g = self.fixedGoal 
        print('Goal set with Coordinates < {}, {} >'.format(g[0], g[1]))
        return g
    
    def _getDist(self):
        return ((self.goal[0] - self.position[0]) ** 2 + (self.goal[1] - self.position[1]) ** 2) ** 0.5

    def _take_action(self, action):
        if self.logging: print('\t\033[1mLinear velocity:\t{}\n\tAngular velocity:\t{}\n\033[0m'.format(action[0], action[1]))
        convertedVelocity = self.setVelocities(action[0], action[1])
        self.motors[0].setVelocity(float(convertedVelocity[0]))
        self.motors[1].setVelocity(float(convertedVelocity[1]))

    def _isDone(self):
        if self.counter >= self.maxTimestep: # Check maximum timestep
            self.needReset = True
            self._endEpisode('timeout')
            return True, 0
        minScan = min(list(filter(lambda a: a != 0, self.lidarRanges[:]))) # Check LiDar
        if minScan < 0.25:
            self.needReset = True
            self._endEpisode('collision')
            return True, -500
        elif self.dist < 0.35: # Check Goal
            self.needReset = True if self.boxEnv else False
            self._endEpisode('success')
            return True, 500
        else:
            return False, 0

    def render(self, mode='human', close=False):
        pass

    def setVelocities(self, linearV, angularV):
        R = 0.033 # Wheel radius
        L = 0.138 # Wheelbase length
        vr = (2 * linearV + angularV * L) / (2 * R)
        vl = (2 * linearV - angularV * L) / (2 * R)
        #print('\nCommanded linear:\t{} angular:\t {}'.format(linearV, angularV))
        #print('Calculated right wheel:\t{}, left wheel:\t {}'.format(vr, vl))
        return [vl, vr]

    def _getDirection(self):
        # Get direction of goal from Robot
        robgoaly = self.goal[1] - self.position[1]
        robgoalx = self.goal[0] - self.position[0]
        goal_angle = math.atan2(robgoalx, robgoaly)
        # Get difference between goal direction and orientation
        heading = goal_angle - self.orientation
        if heading > pi:        # Wrap around pi
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi
        return heading

    def _calculateReward(self):
        action1 = np.round(self.action[0], 1)
        action2 = np.round(self.action[1], 1)
        if self.counter % self.rewardInterval != 0: return 0 # Check if it's Reward Time
        distRate = self.prevDist - self.dist
        if self.prevDist == 0:
            self.prevDist = self.dist
            return 0
        
        moveTowardsGoalReward = self._rewardMoveTowardsGoal(distRate)
        self.moveTowardGoalTotalReward += moveTowardsGoalReward
        
        obsProximityPunish =  self._rewardObstacleProximity()
        self.obsProximityTotalPunish += obsProximityPunish
        
        faceObstaclePunish = -self._rewardFacingObstacle()
        self.faceObstacleTotalPunish += faceObstaclePunish
        
        reward = moveTowardsGoalReward + obsProximityPunish + faceObstaclePunish
        
        #self._printMax(moveTowardsGoalReward)
        totalRewardDic = {"faceObstacleTotalPunish":self.faceObstacleTotalPunish, "obsProximityTotalPunish":self.obsProximityTotalPunish, "moveTowardGoalTotalReward":self.moveTowardGoalTotalReward}
        rewardDic = {"faceObstaclePunish":faceObstaclePunish,"obsProximityPunish":obsProximityPunish,"moveTowardsGoalReward": moveTowardsGoalReward, "Total": reward, "\033[1mTotalReward\033[0m":self.totalreward}
        if self.logging:
            self._printRewards(rewardDic)
            self._printRewards(totalRewardDic)
        self.prevDist = self.dist
        return reward
    
    def _rewardFacingObstacle(self):
        rewardFaceObs = 0
        # Check forward lidars if moving forward, backward lidars if moving backward
        lidarRange = [-2,2] if self.action[0] >= 0 else [7,11]
        for i in range(lidarRange[0],lidarRange[1]):
            scale = 0.15 if i in [-2,1,7,10] else 0.35 #Side readings get less weight
            rewardFaceObs += scale*(1-(self.lidarRanges[i]/self.safetyLimit)) if self.lidarRanges[i] < self.safetyLimit else 0
        return self.faceObstacleParam*rewardFaceObs        
    
    def _rewardMoveTowardsGoal(self, distRate):
        if distRate <= 0:
            return self.moveAwayParam*(distRate/0.0044)
        else:
            return self.moveTowardsParam*(distRate/0.0044)
    
    def _rewardObstacleProximity(self):
        closestObstacle = min(self.lidarRanges)
        if closestObstacle <= self.safetyLimit:
            return  self.obsProximityParam*-(1 - (closestObstacle / self.safetyLimit))
        else:
            return 0

    def SaveAndQuit(self):
        self._write_file('eps.txt', self.currentEpisode)
        with open(self.path + 'P7_NoObstacles.txt', 'a+') as f:
            print('\nSave and Quit \n')
            f.write(repr(self.epInfo))
            f.write('\n')
            f.close()
            print('\tEpisodes Saved')
        self.supervisor.worldReload()

    def _write_file(self, filename, intval):
        with open(self.path + filename, 'w') as fp:
            print(f'Episodes saved: {intval}')
            fp.write(str(intval))
            fp.close()

    def _read_file(self, filename):
        with open(self.path + filename) as fp:
            return int(fp.read())

    def _startEpisode(self):
        if self.start != 0: self._logEpisode()
        # Read episode number
        if self.currentEpisode == 0 and os.path.exists(self.path + 'eps.txt'):
            self.currentEpisode = self._read_file('eps.txt')
        else:
            self.currentEpisode += 1
        print('\n\t\t\t\t EPISODE No. {} \n'.format(self.currentEpisode))
        self.start = time.time()
        if self.needReset: # Reset object(s) / counters / rewards
            print('Resetting Robot...')
            self._resetObject(self.robot, self.robotResetLocation) 
        self.counter = 0
        self.totalreward = 0
        self.moveTowardGoalTotalReward = 0
        self.obsProximityTotalPunish = 0
        self.faceObstacleTotalPunish = 0
        self.prevDist = 0
        
    def _endEpisode(self, ending):
        self.epConc = ending
        prevMode = self.supervisor.simulationGetMode()
        self.supervisor.simulationSetMode(0)
        self.supervisor.exportImage( self.logDir + "screenshots/"+ ending + "/" + str(self.currentEpisode) + ".jpg", 100)
        self.supervisor.simulationSetMode(prevMode)
        print(ending)

    def _getState(self):
        self.lidarRanges = self._trimLidarReadings(self.lidar.getRangeImage())  # Get LidarInfo
        #if self.logging: self._printLidarRanges()
        self.position = self.robot.getPosition()[::2] # Get Position (x,y) and orientation
        self.orientation = self.orientationField.getSFRotation()[3]
        self.dist = self._getDist() # Get Eucledian distance from the goal
        self.direction = self._getDirection()

    def _printRewards(self, r, interval=1):
        if self.counter%interval==0:
            print("\n\033[1mEpisode {}, timestep {}\033[0m".format(self.currentEpisode, self.counter))
            for k, v in r.items():
                print("\t"+ str(k) + ":\t"+ str(v))

    def _printMax(self, reward):
        maxR = reward if self.prevMax < reward else self.prevMax
        self.prevMax = maxR
        print("max: {}".format(maxR))
        
    def _printLidarRanges(self):
        lidarFormatted = []
        for i in range(len(self.lidarRanges)):
            lidarFormatted.append(str(i) + ': ' + str(self.lidarRanges[i]))
        print(lidarFormatted)
            
    def _getLogDirectory(self):
        loggingDir = "Logging/"
        latestFile = ""
        latestMod = 0
        for file in os.listdir(loggingDir):
            fileModTime = os.path.getmtime(loggingDir+file)
            if fileModTime > latestMod:
                latestMod = fileModTime
                latestFile = file
        self.logDir = loggingDir + latestFile + "/"
        self.path = self.logDir + "log/"
            

    def _logEpisode(self):  
        self.duration = time.time() - self.start
        self.epInfo.append([round(self.duration, 3), round(self.totalreward, 3), self.epConc, self.counter,"Goal:", self.goal,"Start position:", self.startPosition, "Rewards:", self.moveTowardGoalTotalReward, self.obsProximityTotalPunish, self.faceObstacleTotalPunish])