from configurator.configuration import *
from model.model import *

# -------------------------------------------参数配置-------------------------------------------
# k值
k = 4
Round = 10
algorithm = 'top-k'
# ----------------------------------------------------------------------------------------------

def eval(model_original, top_k_model, top_ck_model):
    mean_reward_original = []
    std_reward_original = []
    mean_reward_top_k = []
    std_reward_top_k = []
    mean_reward_top_ck = []
    std_reward_top_ck = []

    # 使用 evaluate_policy 函数进行测试评估
    mean_reward, std_reward = evaluate_policy(model_original, env, n_eval_episodes=10, render=True)
    mean_reward_original.append(mean_reward)
    std_reward_original.append(std_reward)
    # 打印测试结果
    # print(f"Original: Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    # 使用 evaluate_policy 函数进行测试评估
    mean_reward, std_reward = evaluate_policy(top_k_model, env, n_eval_episodes=10, render=True)
    mean_reward_top_k.append(mean_reward)
    std_reward_top_k.append(std_reward)
    # 打印测试结果
    # print(f"Compressed: Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    # 使用 evaluate_policy 函数进行测试评估
    mean_reward, std_reward = evaluate_policy(top_ck_model, env, n_eval_episodes=10, render=True)
    mean_reward_top_ck.append(mean_reward)
    std_reward_top_ck.append(std_reward)
    # 打印测试结果
    # print(f"Original: Mean Reward: {mean_reward}, Std Reward: {std_reward}")

    return mean_reward_original, mean_reward_top_k, mean_reward_top_ck

if __name__ == '__main__':
    # env = gym.make(env_name)
    env = gym.make(env_name, render_mode = render_mode)
    env.configure(env_config)
    env.config['duration'] = 50

    model_path = f'{Root_dir}/model/AV_model/highway_dqn/{model_name}'
    model_original = DQN.load(model_file, env=env)

    buffers = dict()

    K = 9
    # for K in tqdm([4, 5, 6, 7, 8, 9, 10]):

    buffers[K] = {}

    top_k_neuron_dir = f"{top_neuron_path}/action5/top-{K}-node-set.data"
    top_ck_neuron_dir = f"{top_neuron_path}/action5/top-c{K}-node-set.data"

    # 关键神经元结点集合
    top_k_sets = deserializer(top_k_neuron_dir)
    top_ck_sets = deserializer(top_ck_neuron_dir)

    top_k_model = compress_model(top_k_sets, env, model_original)
    top_ck_model = compress_model(top_ck_sets, env, model_original)


    # mean_reward_original, mean_reward_top_k, mean_reward_top_ck = eval(model_original, top_k_model, top_ck_model)
    #
    # top_k_neuron_dir = f"{top_neuron_path}/action5/top-{K}-node-set.data"
    # top_ck_neuron_dir = f"{top_neuron_path}/action5/top-c{K}-node-set.data"
    #
    #
    # buffers[K]['top_k_neuron_nums'] = [len(top_k_sets[1]), len(top_k_sets[2])]
    # buffers[K]['top_ck_neuron_nums'] = [len(top_ck_sets[1]), len(top_ck_sets[2])]
    #
    # buffers[K]['mean_reward_original'] = mean_reward_original
    # buffers[K]['mean_reward_top_k'] = mean_reward_top_k
    # buffers[K]['mean_reward_top_ck'] = mean_reward_top_ck

        # # 画折线图
        # plt.plot(duration, mean_reward_original, label='original')
        # plt.plot(duration, mean_reward_compressed, label='compressed')
        #
        # # 添加标题和标签
        # plt.title('Relationship diagram of mean_reward and duration(original model vs compressed model)')
        # plt.xlabel('duration')
        # plt.ylabel('mean_reward')
        #
        # # 添加图例
        # plt.legend()
        #
        # # 显示图形
        # plt.show()

    # serializer(top_neuron_algorithm_test_path, buffers)

    count = 0
    i = 0
    for _ in range(Round):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # action, _states = model_original.predict(obs, deterministic=True)
            action, _states = top_ck_model.predict(obs)
            # if(action == action2):
            #     count = count + 1
            # i = i + 1
            obs, reward, done, truncated, info = env.step(action)
            env.render()

    # print(count/i)


