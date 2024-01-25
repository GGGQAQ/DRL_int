'''test'''

from configurator.configuration import *

# 测试轮次
Round = 5

# Create environment
env = gym.make(env_name, render_mode=render_mode)
env.configure(env_config)
env.config['duration'] = 50
env.reset()

pprint.pprint(env.config)
# load model
model = DQN.load(model_file, env=env)


Test = True


if(Test == True):
    for _ in range(Round):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

else:

    for model_name in model_list:
        model_file = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}/model'
        model_path = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}'
        model = DQN.load(model_file, env=env)
        # 使用evaluate方法评估模型
        mean_reward, episode_length = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True)

        # 打印评估结果
        print(f"Mean reward: {mean_reward}, episode_length: {episode_length}")
        df = pd.DataFrame({'mean_reward':mean_reward, 'episode_length':episode_length})
        df.to_excel(f'{model_path}/evaluate.xlsx', index=False)



