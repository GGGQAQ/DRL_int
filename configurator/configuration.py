'''
整体配置
'''
from utils.utils import *
import gymnasium as gym
from hmmlearn import hmm
import pprint
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import datetime
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import json
import time

import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# algo = 'DQN'
# obs_space = 'Kinematic'
# action_space = 'Meta'

# algo = 'DQN'
# obs_space = 'Grayscale'
# action_space = 'Meta'

# algo = 'PPO'
# obs_space = 'Grayscale'
# action_space = 'Meta'

algo = 'PPO'
obs_space = 'Grayscale'
action_space = 'Continuous'

# algo = 'PPO'
# obs_space = 'Kinematic'
# action_space = 'Continuous'

if (algo == 'DQN' and obs_space == 'Kinematic' and action_space == 'Meta'):
    # 项目根路径
    Root_dir = "D:/0Projects\DRL_based_AV_interpretability"

    # 模型名
    model_name = 'highway-fast-v0env_100000steps_[256, 256]netarch_2023-11-21date'

    # 模型文件
    model_file = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}/model'

    ##模型路径
    model_path = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}'
    # 环境名
    env_name = "highway-fast-modify-v0"
    render_mode = 'rgb_array'

    # 环境配置
    env_config = {
        "collision_reward": -2,
        "high_speed_reward": 0.8,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,  # the default value(rows of observation)
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True, # 进行归一化
            "absolute": False, # 相对坐标
            "order": "sorted", # obs中车辆按照到ego-vehicle的距离远近排序
            "see_behind": False # 看不到后车
        }
    }

    # hmm_data_path
    hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}'

    # 神经网络激活值保存路径
    middle_result_path = f'{Root_dir}/output/middle_result/dqn/{model_name}_3layers_2000rounds'

    # 关键神经元结点保存路径
    top_neuron_path = f"{Root_dir}/output/sampler_result/dqn/{model_name}_3layers_2000rounds"

    # DQN网络模型参数提取路径
    param_path = f'{Root_dir}/output/parameters/{model_name}'

    # 关键神经元提取算法测试结果数据保存路径
    top_neuron_algorithm_test_path = f'{Root_dir}/output/top_neuron_algorithm_test_data/{model_name}'

elif(algo == 'DQN' and obs_space == 'Grayscale' and action_space == 'Meta'):
    # 项目根路径
    Root_dir = "D:/0Projects\DRL_based_AV_interpretability"

    # 模型名
    model_list = ['highway-fast-modify-v0env_100000steps_[64, 16]netarch_2023-12-26date',
                  'highway-fast-modify-v0env_100000steps_[128]netarch_2024-01-09date',
                  'highway-fast-modify-v0env_100000steps_[256]netarch_2023-12-26date']
    model_name = model_list[0]

    # 模型文件
    model_file = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}/model'

    ##模型路径
    model_path = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}'
    # 环境名
    env_name = "highway-fast-modify-v0"
    render_mode = 'rgb_array'

    # 环境配置
    env_config = {
        "collision_reward": -2,
        "high_speed_reward": 0.8,
        "observation": {
            'type': "GrayscaleObservation",
            'observation_shape': (128, 64),
            'stack_size':4,
            'weights': [0.2989, 0.5870, 0.1140],
            'scaling': 1.75,
        },
        'policy_frequency': 2,
        'action': {
            'type': 'DiscreteMetaAction'
        },
    }

    # hmm_data_path
    hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}'

elif(algo == 'PPO' and obs_space == 'Grayscale' and action_space == 'Meta'):
    # 项目根路径
    Root_dir = "D:/0Projects\DRL_based_AV_interpretability"

    # 模型名
    model_name = 'highway-fast-modify-v0env_100000steps_GrayscaleImage_Cnnnetarch_date2023-12-23'

    # 模型文件
    model_file = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}/model'

    # 模型路径
    model_path = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}'
    # 环境名
    env_name = "highway-fast-modify-v0"
    render_mode = 'rgb_array'

    # 环境配置
    env_config = {
        "observation": {
            'type': "GrayscaleObservation",
            'observation_shape': (128, 64),
            'stack_size':4,
            'weights': [0.2989, 0.5870, 0.1140],
            'scaling': 1.75,
        },
        'policy_frequency': 2,
        'action': {
            'type': 'DiscreteMetaAction'
        },
    }

elif(algo == 'PPO' and obs_space == 'Grayscale' and action_space == 'Continuous'):
    # 项目根路径
    Root_dir = "D:/0Projects\DRL_based_AV_interpretability"

    # 模型名
    model_name = 'highway-fast-modify-v0env_1000000steps_GrayscaleImage_netarch[512, 128]_date2024-01-22'

    # 模型文件
    model_file = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}/model'

    ##模型路径
    model_path = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}'
    # 环境名
    env_name = "highway-fast-modify-v0"
    render_mode = 'rgb_array'

    # 环境配置
    env_config = {
        "collision_reward": -2,
        "high_speed_reward": 0.8,
        "observation": {
            'type': "GrayscaleObservation",
            'observation_shape': (128, 64),
            'stack_size':4,
            'weights': [0.2989, 0.5870, 0.1140],
            'scaling': 1.75,
        },
        'offroad_terminal': True, # 车辆出了车道则终止episode，之前ppo训练不好就是因为这里
        'negative_speed_terminal': True,  # 车辆速度为负则终止episode
        'policy_frequency': 2,
        'lane_change_reward': -0.2,
        'action': {
            'type': 'ContinuousAction'
        },
    }

    # hmm_data_path
    hmm_data_path = f'{Root_dir}/output/hmm/ppo/{model_name}'

elif(algo == 'PPO' and obs_space == 'Kinematic' and action_space == 'Continuous'):
    # 项目根路径
    Root_dir = "D:/0Projects\DRL_based_AV_interpretability"

    # 模型名
    model_name = 'highway-fast-modify-v0env_200000steps_Kinematic_netarch[256, 128]_date2024-01-22_2'

    # 模型文件
    model_file = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}/model'

    ##模型路径
    model_path = f'{Root_dir}/model/AV_model/highway_ppo/{model_name}'
    # 环境名
    env_name = "highway-fast-modify-v0"
    render_mode = 'rgb_array'

    # 环境配置
    env_config = {
        "collision_reward": -2,
        "high_speed_reward": 0.8,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
        },
        'offroad_terminal': True, # 车辆出了车道则终止episode，之前ppo训练不好就是因为这里
        'negative_speed_terminal': True, #车辆速度为负则终止episode
        'policy_frequency': 2,
        'lane_change_reward': -0.2,
        'action': {
            'type': 'ContinuousAction'
        },
    }

    # hmm_data_path
    hmm_data_path = f'{Root_dir}/output/hmm/ppo/{model_name}'

else:
    print('false')



