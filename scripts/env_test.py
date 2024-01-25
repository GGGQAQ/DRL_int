import os
import copy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import highway_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import pickle
from find.top_k_finder import Find
from utils.utils import *
from configurator.configuration import *
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from find.register import Register
from model.model import FlattenExtractor, QNetwork, Compressed_QNetwork

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 实验名称
Experiment_name = '20000steps'
# 模型路径
Model_path = f'{Root_dir}/model/AV_model/highway_dqn/{Experiment_name}/model'
# 数据保存路径
Output_path = f'{Root_dir}/output/middle_result/highway_dqn2'
# 测试轮次
Round = 2


if __name__ == '__main__':

    # Create environment
    # env = gym.make("highway-fast-v0", render_mode="rgb_array", max_episode_steps=1000)

    # load model
    # model = DQN.load(f"{Root_dir}/model/AV_model/highway_dqn/20000steps/model", env=env)
    #
    # mean_reward, std_reward = evaluate_policy(
    #     model,
    #     model.get_env(),
    #     deterministic=True,
    #     render=True,
    #     n_eval_episodes=10)
    # print("steps", env.spec.max_episode_steps)
    # print(mean_reward)
    train = True
    if train:
        n_cpu = 1
        batch_size = 64
        seed = 1234
        env = make_vec_env("highway-fast-v0", seed=seed)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    device='cuda:0',
                    tensorboard_log=f"{Root_dir}/model/AV_model/highway_ppo/20000steps/PPO_1")

        # Save the agent
        print(model.policy)
        model.save(f"{Model_path}")