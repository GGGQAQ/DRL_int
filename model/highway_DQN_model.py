import torch as th
from torch import nn
import gymnasium as gym
from stable_baselines3 import DQN

# 创建一个虚拟的 observation_space 和 action_space 对象
observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
action_space = gym.spaces.Discrete(3)

# 自定义的神经网络模型
class CustomNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomNetwork, self).__init__()

        # 获取观测空间的形状
        obs_shape = observation_space.shape

        # 定义网络层
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, action_space.n)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

# 创建虚拟环境
env = gym.make("CartPole-v1")
obs_space = env.observation_space
act_space = env.action_space

# 创建自定义网络
custom_net = CustomNetwork(obs_space, act_space)

# 创建 DQN 模型，并使用自定义网络
model = DQN("MlpPolicy", env, policy_kwargs=dict(features_extractor=custom_net))

# 训练或使用模型
model.learn(total_timesteps=10000)
