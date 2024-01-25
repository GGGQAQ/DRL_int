'''
数据收集，收集<obs, action>
'''
from configurator.configuration import *
from utils.utils import deserializer,serializer
import gymnasium as gym
from stable_baselines3 import DQN, PPO
from find.register import Register
import numpy as np
# import highway_env
import os
from tqdm import tqdm
import pprint
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# --------------------------------------------------------参数配置--------------------------------------------------------
# 环境
env_name = "highspeed-fast-v0"

# 实验名/模型文件
Model_file = 'highspeed-fast-v0env_100000steps_[256, 256]netarch_2023-09-05date_1-highspeedreward'
# 模型路径
Model_path = f"{Root_dir}/model/AV_model/highway_ppo/{Model_file}/model"
# 算法ppo or dqn
Algorithm = 'ppo'
# 输出文件路径
Output_path = f"{Root_dir}/output/obs_act/{Algorithm}/{Model_file}/试一下"

# 测试轮次
Round = 2000

# ------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    env = gym.make(env_name, render_mode='rgb_array')
    env.config['duration'] = 50
    pprint.pprint(env.config)

    # 加载模型
    model = PPO.load(Model_path, env=env)

    buffers = dict()
    # buffers['obs']存放obs
    # buffers['act']存放action
    buffers['obs'] = []
    buffers['act'] = []

    for _ in tqdm(range(Round)):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # print("obs", obs)
            # print(type(obs))
            buffers['obs'].append(obs)

            # Predict
            action, _states = model.predict(obs, deterministic=True)

            buffers['act'].append(action)

            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done or truncated:
                print("round:", _)
                print("info:", info)

    print("obslen", len(buffers['obs']))
    print("actlen", len(buffers['act']))

    serializer(f"{Output_path}/obs.data", buffers['obs'])
    print("存储成功！")

    # buffers['act'] = np.concatenate(buffers['act'], axis=0)
    serializer(f"{Output_path}/act.data", buffers['act'])
    print("存储成功！")



