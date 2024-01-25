from configurator.configuration import *
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import datetime
# import highway_env
import pprint
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':

    env = gym.make("highway-v0", render_mode='rgb_array')

    env.configure({
        "manual_control": True
    })


