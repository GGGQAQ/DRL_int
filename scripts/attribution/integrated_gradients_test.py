# 给定参数（指定某一层），求出所有日志数据的该层的积分梯度归因值存放到文件attr.data，以及对应的label存放到label.data，以便后续进行聚类验证是否有明显区分效果
# 5w条数据约50min

import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import *
from configurator.configuration import *
import gymnasium as gym
from stable_baselines3 import DQN, PPO
import numpy as np
from attribution.intergrated_gradients import *

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
# ------------------------------------------------------------------------------------------------------------------


def copy_net(model):
    if(Algorithm == 'ppo'):
        net = nn.Sequential(nn.Flatten(start_dim=0, end_dim=-1),
                            model.policy.mlp_extractor.policy_net[0],
                            model.policy.mlp_extractor.policy_net[1],
                            model.policy.mlp_extractor.policy_net[2],
                            model.policy.mlp_extractor.policy_net[3],
                            model.policy.action_net)
        '''
        net结构：
        Sequential(
            (0): Flatten(start_dim=1, end_dim=-1)
            (1): Linear(in_features=25, out_features=256, bias=True)
            (2): Tanh()
            (3): Linear(in_features=256, out_features=256, bias=True)
            (4): Tanh()
            (5): Linear(in_features=256, out_features=5, bias=True)
            )
        '''
    else:
        net = nn.Sequential(nn.Flatten(start_dim=0, end_dim=-1),
                            model.policy.q_net.q_net[0],
                            model.policy.q_net.q_net[1],
                            model.policy.q_net.q_net[2],
                            model.policy.q_net.q_net[3],
                            model.policy.q_net.q_net[4])
        '''
        net结构：
        Sequential(
            (0): Flatten(start_dim=1, end_dim=-1)
            (1): Linear(in_features=25, out_features=256, bias=True)
            (2): ReLU()
            (3): Linear(in_features=256, out_features=256, bias=True)
            (4): ReLU()
            (5): Linear(in_features=256, out_features=5, bias=True)
            )
        '''
    return net


if __name__ == '__main__':

    start_time = datetime.datetime.now()
    env = gym.make(Env_name)

    # 加载模型
    if(Algorithm == 'ppo'):
        model = PPO.load(Model_path, env=env)
    else:
        model = DQN.load(Model_path, env=env)

    net = copy_net(model)

    obs = deserializer(Data_path)

    # 网络结构提取
    net_01 = nn.Sequential(net[0], net[1])
    net_23 = nn.Sequential(net[2], net[3])
    net_2345 = nn.Sequential(net[2], net[3], net[4], net[5])
    net_45 = nn.Sequential(net[4], net[5])

    # 积分梯度算法用到的基线值，输入为0，然后经过各层时对应各层的基线值
    baseline = np.zeros_like(obs[0])
    baseline = torch.from_numpy(baseline).to("cuda")
    baseline_01 = net_01(baseline)
    baseline_0123 = net_23(baseline_01)

    # 存放归因结果
    attr_01 = list()
    attr_0123 = list()

    for _ in tqdm(range(len(obs))):
        input = torch.from_numpy(obs[_]).to("cuda")
        input_01 = net_01(input)
        input_0123 = net_23(input_01)
        integrated_gradient_01 = integrated_gradients(net_2345, input_01, baseline_01)
        integrated_gradient_0123 = integrated_gradients(net_45, input_0123, baseline_0123)
        attr_01.append(integrated_gradient_01)
        attr_0123.append(integrated_gradient_0123)

    output_file_01 = os.path.join(Output_path, 'attr_01.data')
    output_file_0123 = os.path.join(Output_path, 'attr_0123.data')

    serializer(output_file_01,attr_01)
    serializer(output_file_0123, attr_0123)

    print(net)

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print("开始时间:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("结束时间:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"执行时间: {execution_time}")


