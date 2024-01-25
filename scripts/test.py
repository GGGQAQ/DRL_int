import torch
import gymnasium as gym
import torch.nn as nn
import torch
from utils.utils import *
from configurator.configuration import *
from stable_baselines3 import DQN
from tqdm import tqdm
# --------------------------------------------------------参数配置--------------------------------------------------------
# 环境
env_name = "highspeed-fast-v0"
# 实验名/模型文件
Model_file = 'highspeed-fast-v0env_20000steps_GrayscaleImage_Cnnnetarch_2023-10-23date'
# 模型路径
Model_path = f"{Root_dir}/model/AV_model/highway_dqn/{Model_file}/model"
# 算法ppo or dqn
Algorithm = 'dqn'
# 输出文件路径
Output_path = f"{Root_dir}/output/obs_act/{Algorithm}/{Model_file}"
# 测试轮次
Round = 2000
# 测试数据路径
data_path = 'D:/0Projects\DRL_based_AV_interpretability\output\obs_act\dqn\highspeed-fast-v0env_20000steps_GrayscaleImage_Cnnnetarch_2023-10-23date/'
# 输出路径
output_path = f'{Root_dir}/output/middle_result/dqn/{Model_path}/'
# ------------------------------------------------------------------------------------------------------------------------

# env = gym.make(env_name, render_mode='rgb_array')
env = gym.make(env_name)
config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 1,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2
}
env.configure(config)
env.reset()
env.config['duration'] = 50

# 加载模型
model = DQN.load(Model_path, env=env)


# 创建一个新的 Sequential 网络
new_sequential_model = nn.Sequential()

# 将 features_extractor 添加到新的 Sequential 网络
new_sequential_model.add_module('features_extractor', model.q_net.features_extractor)

# 将 q_net 添加到新的 Sequential 网络
new_sequential_model.add_module('q_net', model.q_net.q_net)

# 打印新的 Sequential 网络
print(new_sequential_model)