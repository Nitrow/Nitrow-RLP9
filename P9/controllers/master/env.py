#!/home/mark/PycharmProjects/P7_openAI/Testing/bin/python
# from stable_baselines import PPO2
import numpy as np
import gym
from gym import spaces
from P7_driver import Driver

class weBotsEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(weBotsEnv, self).__init__()

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self, sv):
        # Get the stock data points for the last 5 days and scale to between 0-1
        obs = sv.lidar.getRangeImage()
        print(obs)
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        pass

    def step(self, action):
        # Execute one time step within the environment
       
        delay_modifier = 2

        reward = 2 * delay_modifier
        done = -1 <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset2(self, lid):
        print(lid)        

    def reset(self, lid):
        # Reset the state of the environment to an initial state
        

        # Set the current step to a random point within the data frame
        print(lid)


    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
