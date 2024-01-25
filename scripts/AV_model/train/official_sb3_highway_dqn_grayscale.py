from configurator.configuration import *

# -------------------------------------------参数配置-------------------------------------------
# 网络结构
net_arch = [128]
# 日期
today = datetime.date.today()
# 迭代次数
Total_timesteps = int(1e5)
# 实验命名
Experiment_name = f'{env_name}env_{Total_timesteps}steps_{net_arch}netarch_{today}date'
# 日志路径
Logdir = f'{Root_dir}/model/AV_model/highway_dqn/{Experiment_name}/'

# ----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    env = gym.make(env_name)
    env.configure(env_config)
    env.reset()

    model = DQN('CnnPolicy', env,
                  policy_kwargs=dict(net_arch=net_arch),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  target_update_interval=50,
                  verbose=1,
                  tensorboard_log=Logdir)

    print(model.policy.q_net)

    model.learn(total_timesteps=Total_timesteps)
    model_file = f'{Root_dir}/model/AV_model/highway_dqn/{Experiment_name}/model'
    model.save(model_file)




