import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN

from stable_baselines3.common.vec_env import DummyVecEnv

env_name = 'highway-v0'
env = gym.make(env_name, render_mode = "rgb_array")


# 定义一个简单的CNN模型
class CustomCNNPolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x

policy_kwargs = dict(
    features_extractor_class=CustomCNNPolicy,
    features_extractor_kwargs={}
)

model = DQN('CnnPolicy', DummyVecEnv([lambda: env]), policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=100)

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
        break


