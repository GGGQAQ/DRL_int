from stable_baselines3 import PPO, DQN
import gymnasium as gym
import torch
from configurator.configuration import *
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import os
from utils.utils import *
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------------------------------参数配置-----------------------------------------------------
# 实验环境
Env_name = 'highspeed-fast-v0'
# 模型文件
Model_file = 'highspeed-fast-v0env_100000steps_[256, 256]netarch_2023-09-05date_1-highspeedreward'

# 模型路径
Model_path = f'{Root_dir}\model\AV_model\highway_ppo\{Model_file}\model.zip'
# DQN or PPO ppo和dqn的policy结构不同，区别处理
Algorithm = 'ppo'

# <obs, act>存放路径
Data_path = f'{Root_dir}\output\obs_act\ppo\{Model_file}/obs.data'

# 实验输出路径
Output_path = f'{Root_dir}\output/attribution_result\ppo\{Model_file}'

Attr_path = f'{Root_dir}\output/attribution_result\ppo\{Model_file}'
# ------------------------------------------------------------------------------------------------------------------



attr_01_path = os.path.join(Attr_path, 'attr_0.data')
attr_01 = deserializer(attr_01_path)
attr_01 = attr_01[:100]

print(type(attr_01))
print(attr_01[0].cpu().detach().numpy())

attr_01 = [tensor.cpu().detach().numpy() for tensor in attr_01]
# 创建一个包含 list 中所有数据的 NumPy 数组

attr_01 = np.array(attr_01)

# 创建 x 轴的标签，表示 list 中每个元素的下标
x_labels = np.arange(attr_01.shape[1])

print("len", len(attr_01))

# 绘制点状分布图
plt.figure(figsize=(10, 6))
for data in attr_01:
    plt.scatter(x_labels, data)

# 设置图表标题和标签
# plt.title('所有数据的点状分布图')
# plt.xlabel('数据点索引')
# plt.ylabel('取值')

# 显示图表
plt.show()