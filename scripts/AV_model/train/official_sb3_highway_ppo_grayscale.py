import pprint

from configurator.configuration import *

# -------------------------------------------参数配置-------------------------------------------
# 网络结构
net_arch = [512, 128]
# 日期
today = datetime.date.today()
# 迭代次数
Total_timesteps = int(1e5)
# 实验命名
Experiment_name = f'{env_name}env_{Total_timesteps}steps_GrayscaleImage_netarch{net_arch}_date{today}'
# 日志路径
Logdir = f'{Root_dir}/model/AV_model/highway_ppo/{Experiment_name}/'
# 模型保存路径，位于日志路径下
Model_path = f'{Logdir}/model'
# ----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Train
    env = gym.make(env_name)
    env.configure(env_config)
    env.reset()

    policy_kwargs = dict(net_arch=[dict(pi=net_arch, vf=net_arch)])
    model = PPO('CnnPolicy', env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=Logdir)

    print(model.policy)

    pprint.pprint(env.config)

    model.learn(total_timesteps=Total_timesteps)
    model.save(Model_path)

