import pprint

import model.model_loader
from configurator.configuration import *

# -------------------------------------------参数配置-------------------------------------------
# 网络结构
net_arch = [512, 128]
# 日期
today = datetime.date.today()
# 迭代次数
Total_timesteps = int(1e6)
# 实验命名
Experiment_name = f'{env_name}env_{Total_timesteps}steps_GrayscaleImage_netarch{net_arch}_date{today}'
# 日志路径
model_path = f'{Root_dir}/model/AV_model/highway_ppo/{Experiment_name}'
# 模型保存路径，位于日志路径下
model_file = f'{model_path}/model'
# ----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    start_time = time.time()

    # Train
    env = gym.make(env_name)
    env.configure(env_config)
    env.reset()

    policy_kwargs = dict(net_arch=[dict(pi=net_arch, vf=net_arch)])
    model = PPO('CnnPolicy', env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=model_path)

    # model = PPO.load(model_file, env)
    pprint.pprint(env.config)

    model.learn(total_timesteps=Total_timesteps)
    model.save(model_file)

    # 将字典保存到文件
    with open(f'{model_path}/config.txt', 'w') as file:
        json.dump(env.config, file, indent=4)

    end_time = time.time()
    print_time(start_time, end_time)