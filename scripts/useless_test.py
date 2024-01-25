import gymnasium

from configurator.configuration import *
import pprint

import torchvision.models as models


    # # 保存图形到文件，可以指定文件格式（如png、jpg、pdf等）
    # plt.savefig(f'{Root_dir}/picture_{_}.png')


import numpy as np


# 假设你有一个包含数据的列表 data_list
# data_list 中的每个元素是一个 shape 为 (1, 67) 的 NumPy 数组

# K = 4
# test_model_name = f'top_c{K}_model'
# hmm_data_path = f'{Root_dir}/output/hmm/dqn/{model_name}/{test_model_name}'
# data_list = deserializer(f"{hmm_data_path}/activation3.data")
#
# # 将数据整合成一个二维 NumPy 数组
# print(data_list[0].shape)
# print(data_list[0])
# data_array = np.vstack(data_list)
#
#
# # 创建 PCA 模型，指定降维后的维度为 40
# pca = PCA(n_components=40)
#
# # 用数据拟合 PCA 模型
# pca.fit(data_array)
#
# # 对数据应用 PCA 变换，降维到 40 维
# data_pca = pca.transform(data_array)
#
#
# data_pca_list = data_pca.tolist()
# new_list = list()
# for _ in data_pca_list:
#     new_list.append(np.array(_))
#
# print(new_list[0].shape)


# import matplotlib.pyplot as plt
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# # 训练模型
# model.learn(total_timesteps=10000)
#
# # 保存模型
# model.save("ppo_cartpole")
#
# # 关闭环境
# env.close()

# 加载已训练的模型
# loaded_model = PPO.load("ppo_cartpole")

# # 在环境中应用模型
# obs, info = env.reset()
# for _ in range(1000):
#     action, states = loaded_model.predict(obs, deterministic=True)
#     obs, re, done, tr, info = env.step(action)
#     env.render()
#     if done:
#         obs, info = env.reset()

import torch
# import torchvision.models as models
#
# # 加载预训练的 AlexNet 模型
# alexnet = models.alexnet(pretrained=False)
#
# print(alexnet)

#
# env = gym.make(env_name, render_mode='rgb_array')
# env = gym.make(env_name)
# env.configure(env_config)
# env.config['duration'] = 50
# env.reset()
# #
# model = DQN.load(model_file, env=env)
#
# print(model.policy.q_net)




# import torch
# from torch import nn
# from torchviz import make_dot
# from torchvision.models import vgg16
#
#
#
# # # 加载预训练的 VGG16 模型
# model = vgg16(pretrained=False)
#
# # 获取一个示例输入
# example_input = torch.rand(128, 512)
#
# # 使用 torchviz 创建计算图
# output = model(example_input)
# dot = make_dot(output, params=dict(model.named_parameters()))
#
# # 保存计算图为 PDF 文件
# dot.format = 'pdf'
# dot.render("vgg16_graph")




# 画一下分布图

# import seaborn as sns
# for layer in [5]:
#     data = deserializer(f'{hmm_data_path}/activation{layer}.data')
#     print(type(data[0]))
#     for _ in range(data[0].shape[1]):
#         result_array = np.array([arr[0][1] for arr in data])
#         APoZ = non_zero_count = np.count_nonzero(result_array) / result_array.shape[0]
#         print(f"{APoZ}")
#
#         # # 使用 seaborn 绘制 KDE 图
#         # sns.set(style="whitegrid")  # 设置 seaborn 样式
#         # sns.kdeplot(result_array, color="blue", fill=True)
#         #
#         # # 添加标签和标题
#         # plt.xlabel('X-axis')
#         # plt.ylabel('Density')
#         # plt.title(f'Kernel Density Estimation (KDE) of layer{layer}')
#         #
#         # # 显示图形
#         # plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# # 生成数据
# mean1, std1 = 38.58, 25.98  # 第一个正态分布的均值和标准差
# mean2, std2 = 75.32, 28.16  # 第二个正态分布的均值和标准差
#
# x = np.linspace(-10, 140, 1000)  # 生成 x 值
#
# # 生成正态分布曲线的 y 值
# y1 = norm.pdf(x, mean1, std1)
# y2 = norm.pdf(x, mean2, std2)
#
# # 绘制曲线
# plt.plot(x, y1, label='Normal Distribution 1')
# plt.plot(x, y2, label='Normal Distribution 2')
#
# # 添加标题和标签
# plt.title('Two Normal Distributions')
# plt.xlabel('X-axis')
# plt.ylabel('Probability Density Function')
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()

# 不同模型效果对比
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
#
# data = list()
# for _, model_name in enumerate(model_list):
#     model_path = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}'
#     data.append(pd.read_excel(f'{model_path}/evaluate.xlsx')['mean_reward'])
#
#
# # 绘制三组数据的分布曲线
# for i in range(3):
#     kde = gaussian_kde(data[i])
#     # x_vals = np.linspace(min(data[i]), max(data[i]), 150)
#     x_vals = np.linspace(-10, 140, 140)
#     plt.plot(x_vals, kde(x_vals), label=f'Distribution {i+1}')
#
#     # 添加均值和标准差的垂直线和文本标注
#     mean_value = np.mean(data[i])
#     std_deviation = np.std(data[i])
#     plt.axvline(mean_value, color=f'C{i}', linestyle='dashed', linewidth=1, label=f'Mean {i + 1}')
#     plt.text(mean_value, 0.03 + i * 0.02, f'Mean: {mean_value:.2f}\nStd: {std_deviation:.2f}', color=f'C{i}',
#              ha='center')
#
# # 添加标题和标签
# plt.title('Distributions')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
#
# # 添加图例
# plt.legend()
#
# # 显示图形
# plt.show()

# 画一下分布图

# import seaborn as sns
# for layer in [5]:
#     data = deserializer(f'{hmm_data_path}/activation{layer}.data')
#     print(type(data[0]))
#     for _ in range(data[0].shape[1]):
#         result_array = np.array([arr[0][1] for arr in data])
#         APoZ = non_zero_count = np.count_nonzero(result_array) / result_array.shape[0]
#         print(f"{APoZ}")
#
#         # 使用 seaborn 绘制 KDE 图
#         sns.set(style="whitegrid")  # 设置 seaborn 样式
#         sns.kdeplot(result_array, color="blue", fill=True)
#
#         # 添加标签和标题
#         plt.xlabel('X-axis')
#         plt.ylabel('Density')
#         plt.title(f'Kernel Density Estimation (KDE) of layer{layer}')
#
#         # 显示图形
#         plt.show()

# import numpy as np
#
# # 假设样本存放在一个名为 samples 的 list 中
# samples = [np.random.randint(0, 2, size=(1, 512)) for _ in range(10000)]  # 生成随机样本作为示例
#
# # 将样本堆叠成一个大的数组，形状为 (样本数量, 512)
# stacked_samples = np.vstack(samples)
#
# # 统计每一位变量为0的概率，即计算每列中0的个数除以样本数量
# zero_probabilities = np.mean(stacked_samples == 0, axis=0)
#
# print("每一位变量为0的概率:", zero_probabilities)

# # -------------------------------------------参数配置-------------------------------------------
# # 网络结构
# net_arch = 'Cnn'
# # 日期
# today = datetime.date.today()
# # 迭代次数
# Total_timesteps = int(100)
# # 实验命名
# Experiment_name = f'{env_name}env_{Total_timesteps}steps_GrayscaleImage_{net_arch}netarch_date{today}'
# # 日志路径
# Logdir = f'{Root_dir}/model/AV_model/highway_ppo/{Experiment_name}/'
# # 模型保存路径，位于日志路径下
# Model_path = f'{Logdir}/model'
# # ----------------------------------------------------------------------------------------------
#
# if __name__ == '__main__':
#     # Train
#     env = gym.make(env_name)
#     env.configure(env_config)
#     env.reset()
#
#     policy_kwargs = dict(net_arch=[dict(pi=[512], vf=[512])])
#     model = PPO('CnnPolicy', env,
#                 policy_kwargs=policy_kwargs,
#                 verbose=1,
#                 tensorboard_log=Logdir)
#
#     print(model.policy)
#
#     pprint.pprint(env.config)
#
#     model.learn(total_timesteps=Total_timesteps)
#     model.save(Model_path)

# import time
# # from datetime import datetime
# # 记录开始时间
# start_time = time.time()
#
# formatted_date = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
# print(f"开始时间: {formatted_date}")
#
# # 在这里放置您要测试耗时的代码块
# for i in range(1000000):
#     _ = i * i
#
# # 记录结束时间
# end_time = time.time()
#
# # 计算执行时间
# elapsed_time = end_time - start_time
#
# # 将总秒数转换为时、分、秒
# hours, remainder = divmod(elapsed_time, 3600)
# minutes, seconds = divmod(remainder, 60)
#
# print(f"代码执行耗时：{int(hours)} 小时 {int(minutes)} 分钟 {seconds:.2f} 秒")

import gymnasium
import torch as th
from stable_baselines3 import PPO
from sb3_contrib import TQC
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
import warnings
warnings.filterwarnings('ignore')

# ==================================
#        Main script
# ==================================


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], render_mode = 'rgb_array')
    env.configure(kwargs["config"])
    env.reset()
    return env


env_kwargs = {
    'id': 'highway-fast-v0',
    'config': {
        "action": {
            "type": "ContinuousAction"
        }
    }
}

# n_cpu = 6
# batch_size = 64
env = make_configure_env(**env_kwargs)
env.reset()
policy_kwargs = dict(n_critics=2, n_quantiles=25)
train=False
if train:
    model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=3000)
    # Save the agent
    model.save("highway_tqc/model_v2")
model=TQC.load("highway_tqc/model_v2")
env = make_configure_env(**env_kwargs)
env.reset()
for _ in range(50):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, tr, info = env.step(action)
        env.render()














